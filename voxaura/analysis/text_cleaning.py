"""Transcript cleaning: filler-word detection/removal and abbreviation expansion.

The word lists live in config/fillers.json and config/abbreviations.json so they
can be changed without touching code.

Fillers come in three kinds:
  * hesitations ("um", "uh") - always fillers;
  * discourse markers ("like", "you know", "I mean", "kind of") - fillers only in
    certain contexts. Each phrase has a rule that looks at the words around it,
    e.g. "So like we should go" (filler) versus "I like pizza" (verb);
  * repetitions - a word said twice in a row, e.g. "the the results".

Abbreviations come in two kinds:
  * chat abbreviations ("idk", "btw") are replaced by their meaning;
  * acronyms ("AI", "U.S.") are kept and explained on first use:
    "AI (artificial intelligence)".
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..ingestion.transcription import Segment, split_sentences

# A token must not be glued to letters, digits, dots, apostrophes, hyphens or "&"
# on either side. This keeps "um" out of "umbrella", "uh" out of "uh-huh", and
# "U" / "US" out of "U.S." or "R&D".
_BEFORE = r"(?<![\w.'&-])"
_AFTER = r"(?![\w'&-]|\.\w)"

REPETITION = "repeated word"

# Word lists used by the discourse-marker rules below.
CONJUNCTIONS = {"so", "and", "but", "or", "because", "cause", "then", "yeah", "oh", "well", "plus",
                "also", "now", "okay"}
BE_WORDS = {"was", "were", "is", "are", "am", "be", "been", "it's", "that's", "he's", "she's",
            "i'm", "you're", "we're", "they're"}
INTENSIFIERS = {"really", "so", "very", "totally", "super", "just", "literally", "completely",
                "actually", "kind", "sort"}
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "fifteen", "twenty", "thirty", "forty", "fifty", "hundred",
                "thousand", "million", "half"}
LIKE_KEEP_PREV = {"i", "you", "we", "they", "he", "she", "would", "i'd", "you'd", "we'd", "they'd",
                  "don't", "didn't", "do", "does", "doesn't", "really", "also", "to", "look", "looks",
                  "looked", "looking", "seem", "seems", "seemed", "feel", "feels", "felt", "sound",
                  "sounds", "sounded", "something", "nothing", "anything", "just", "more", "much",
                  "not", "people", "things", "lot"}
LIKE_KEEP_NEXT = {"to", "that", "this", "these", "those", "it", "him", "her", "them", "me", "us",
                  "you", "a", "an", "the", "what", "how", "when", "my", "your", "his", "our", "their"}
YOU_KNOW_KEEP_PREV = {"do", "did", "does", "don't", "didn't", "doesn't", "if", "as", "what", "how",
                      "would", "will", "i", "you", "they", "we", "he", "she", "let", "may", "might",
                      "could", "should", "never", "to", "who", "whether", "must", "already"}
YOU_KNOW_KEEP_NEXT = {"what", "how", "why", "where", "who", "when", "which", "that", "if", "whether",
                      "about", "him", "her", "them", "me", "us", "this", "these", "those", "someone",
                      "somebody", "anything", "anyone", "everything", "everyone", "nothing", "the",
                      "a", "an", "my", "your", "his", "our", "their", "more", "enough", "better"}
I_MEAN_KEEP_NEXT = {"that", "what", "to", "no", "business", "you", "him", "her", "them", "me", "this",
                    "very", "exactly", "every", "literally", "seriously", "by"}
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "what", "which", "some", "any",
               "every", "each", "one", "same", "different", "other", "all", "no", "my", "your",
               "his", "her", "our", "their", "its", "another", "whatever", "certain", "only", "new"}


def _alternation(words) -> str:
    # Longest first, so "you know what I mean" wins over "you know".
    return "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))


@dataclass
class FillerStats:
    counts: dict[str, int] = field(default_factory=dict)
    total_words: int = 0

    @property
    def total_fillers(self) -> int:
        return sum(self.counts.values())

    @property
    def per_100_words(self) -> float:
        if not self.total_words:
            return 0.0
        return round(100 * self.total_fillers / self.total_words, 2)

    def to_dict(self) -> dict:
        return {"counts": self.counts, "total_words": self.total_words}

    @classmethod
    def from_dict(cls, data: dict | None) -> "FillerStats":
        data = data or {}
        return cls(counts=data.get("counts", {}), total_words=data.get("total_words", 0))


@dataclass
class FillerMatch:
    start: int
    end: int
    label: str  # the filler as counted, e.g. "um", "you know", "repeated word"


@dataclass
class _Context:
    """What surrounds a candidate phrase - the input to every discourse-marker rule."""

    prev_word: str | None  # word directly before (None if punctuation or start of text)
    next_word: str | None  # word directly after (None if punctuation or end of text)
    next_word_ends_clause: bool  # the next word is followed by punctuation or the end
    clause_start: bool  # start of text, after punctuation, or after a conjunction / hesitation
    after_comma: bool  # directly preceded by a comma
    comma_after: bool  # directly followed by a comma
    end_after: bool  # followed by . ! ? ; : or the end of the text

    @property
    def pause_after(self) -> bool:
        return self.comma_after or self.end_after


def _context(text: str, start: int, end: int, hesitations: set[str]) -> _Context:
    before = text[:start].rstrip()
    after = text[end:]
    prev = re.search(r"([\w']+)$", before)
    prev_word = prev.group(1).lower() if prev else None
    nxt = re.match(r"\s+([\w']+)", after)
    next_word = nxt.group(1).lower() if nxt else None
    rest = after[nxt.end():] if nxt else ""
    at_boundary = not before or before[-1] in ".!?;:,—–(\"“"
    return _Context(
        prev_word=prev_word,
        next_word=next_word,
        next_word_ends_clause=bool(nxt) and (not rest.strip() or rest[:1] in ",.!?;:"),
        clause_start=at_boundary or prev_word in CONJUNCTIONS or prev_word in hesitations,
        after_comma=before.endswith(","),
        comma_after=after.startswith(","),
        end_after=not after.strip() or after[:1] in ".!?;:",
    )


def _is_number(word: str | None) -> bool:
    return bool(word) and (word.isdigit() or word in NUMBER_WORDS)


# --- Discourse-marker rules: each returns True if the phrase is a filler here ---

def _rule_like(c: _Context) -> bool:
    if c.prev_word in LIKE_KEEP_PREV:  # "I like", "looks like", "just like"
        return False
    if c.clause_start and c.comma_after:  # "It was, like, huge" / "Like, it was"
        return True
    if c.next_word in {"um", "uh", "er", "erm"}:  # "like um"
        return True
    if c.prev_word in CONJUNCTIONS and c.next_word and c.next_word not in LIKE_KEEP_NEXT:
        return True  # "So like we should", "and like everyone"
    if c.prev_word in BE_WORDS and (c.comma_after or c.next_word in INTENSIFIERS):
        return True  # "I was like really tired"
    return _is_number(c.next_word) and c.prev_word is not None  # "there were like 20 people"


def _rule_you_know(c: _Context) -> bool:
    if c.prev_word in YOU_KNOW_KEEP_PREV or c.next_word in YOU_KNOW_KEEP_NEXT:
        return False  # "do you know", "you know what happened"
    if c.next_word == "it" and c.next_word_ends_clause:
        return False  # "you know it."
    return c.clause_start or c.pause_after


def _rule_i_mean(c: _Context) -> bool:
    if c.comma_after or (c.after_comma and c.end_after):
        return True  # "I mean, it was..." / "..., I mean."
    if not c.clause_start or c.next_word is None or c.next_word in I_MEAN_KEEP_NEXT:
        return False  # "I mean what I say"
    return not (c.next_word == "it" and c.next_word_ends_clause)  # "I mean it."


RULES = {
    "always": lambda c: True,
    "like": _rule_like,
    "you_know": _rule_you_know,
    "i_mean": _rule_i_mean,
    # "Actually, ..." / "so, honestly, ..." - must be followed by a comma
    "comma": lambda c: c.clause_start and c.comma_after,
    # "Basically we need..." - at the start of a clause
    "clause_start": lambda c: c.clause_start and (c.next_word is not None or c.comma_after),
    # "... or something." - at the end of a clause
    "clause_end": lambda c: c.pause_after,
    # "..., right?" - a tag after a comma at the end of a sentence
    "tag": lambda c: c.after_comma and c.end_after,
    # "it was kind of good" but not "this kind of car"
    "hedge": lambda c: c.prev_word not in DETERMINERS and c.next_word is not None,
    # "it was fine, I guess." / "I guess we should..."
    "edge": lambda c: c.pause_after or (c.clause_start and c.next_word not in {None, "so", "that", "what", "not"}),
}


class TextCleaner:
    def __init__(self, fillers: list[str], abbreviations: dict[str, str],
                 contextual_fillers: list[str] | None = None, acronyms: dict[str, str] | None = None,
                 discourse_markers: dict[str, str] | None = None, allowed_repeats: list[str] | None = None,
                 detect_repetitions: bool = False):
        self.fillers = [f.lower() for f in fillers]
        self.discourse_markers = {k.lower(): v for k, v in (discourse_markers or {}).items()}
        for phrase in contextual_fillers or []:  # older config format: comma rule
            self.discourse_markers.setdefault(phrase.lower(), "comma")
        unknown = set(self.discourse_markers.values()) - set(RULES)
        if unknown:
            raise ValueError(f"Unknown filler rule(s) in config: {', '.join(sorted(unknown))}")
        self.detect_repetitions = detect_repetitions
        self.allowed_repeats = {w.lower() for w in allowed_repeats or []}
        self.abbreviations = {k.lower(): v for k, v in abbreviations.items()}
        self.acronyms = dict(acronyms or {})

        self._hesitation_re = (
            re.compile(_BEFORE + "(" + _alternation(self.fillers) + ")" + _AFTER, re.IGNORECASE)
            if self.fillers else None
        )
        self._discourse_re = (
            re.compile(r"(?<![\w'])(" + _alternation(self.discourse_markers) + r")(?![\w'])", re.IGNORECASE)
            if self.discourse_markers else None
        )
        self._repeat_re = re.compile(r"(?<![\w'])([A-Za-z']+)((?:,?\s+\1)+)(?![\w'])", re.IGNORECASE)
        self._abbr_re = (
            re.compile(_BEFORE + "(" + _alternation(self.abbreviations) + ")" + _AFTER, re.IGNORECASE)
            if self.abbreviations else None
        )
        # Acronyms are case-sensitive: "US" is the country, "us" is a pronoun.
        self._acronym_re = (
            re.compile(r"(?<![\w.&-])(" + _alternation(self.acronyms) + r")(?![\w&-]|(?<!\.)\.\w)")
            if self.acronyms else None
        )

    @classmethod
    def from_config_dir(cls, config_dir: Path) -> "TextCleaner":
        with open(config_dir / "fillers.json", encoding="utf-8") as f:
            fillers = json.load(f)
        with open(config_dir / "abbreviations.json", encoding="utf-8") as f:
            abbreviations = json.load(f)
        if "chat_abbreviations" not in abbreviations:  # older flat format
            abbreviations = {"chat_abbreviations": abbreviations}
        repetitions = fillers.get("repetitions", {})
        return cls(
            fillers=fillers.get("hesitations", fillers.get("fillers", [])),
            contextual_fillers=fillers.get("contextual_fillers", []),
            discourse_markers=fillers.get("discourse_markers", {}),
            detect_repetitions=repetitions.get("enabled", False),
            allowed_repeats=repetitions.get("allowed_repeats", []),
            abbreviations=abbreviations["chat_abbreviations"],
            acronyms=abbreviations.get("acronyms", {}),
        )

    # ------------------------------ fillers ------------------------------ #

    def find_fillers(self, text: str) -> list[FillerMatch]:
        """All filler occurrences in the text, in order and without overlaps."""
        found: list[FillerMatch] = []
        hesitations = set(self.fillers)
        if self._hesitation_re:
            found += [FillerMatch(m.start(1), m.end(1), m.group(1).lower())
                      for m in self._hesitation_re.finditer(text)]
        if self._discourse_re:
            for m in self._discourse_re.finditer(text):
                phrase = m.group(1).lower()
                if RULES[self.discourse_markers[phrase]](_context(text, m.start(1), m.end(1), hesitations)):
                    found.append(FillerMatch(m.start(1), m.end(1), phrase))
        if self.detect_repetitions:
            for m in self._repeat_re.finditer(text):
                word = m.group(1).lower()
                if word not in self.allowed_repeats and word not in hesitations:
                    # the first occurrence stays; the repeat(s) are the filler
                    found.append(FillerMatch(m.end(1), m.end(2), REPETITION))
        found.sort(key=lambda f: (f.start, -(f.end - f.start)))
        result: list[FillerMatch] = []
        for f in found:
            if not result or f.start >= result[-1].end:
                result.append(f)
        return result

    def detect_fillers(self, text: str) -> FillerStats:
        counts = Counter(f.label for f in self.find_fillers(text))
        return FillerStats(counts=dict(counts.most_common()), total_words=len(text.split()))

    def remove_fillers(self, text: str) -> str:
        """Remove fillers together with the commas that set them off."""
        cuts: list[list[int]] = []
        question_to_period: set[int] = set()
        for f in self.find_fillers(text):
            start, end = f.start, f.end
            if f.label != REPETITION:  # a repetition span already includes its leading space
                before, after = text[:start], text[end:]
                if after.startswith(","):  # "were, um, better" -> "were better"
                    end += 1
                    if before.endswith(", "):
                        start -= 2
                elif before.endswith(", ") and (not after.strip() or after[:1] in ".!?"):
                    start -= 2  # "it was hard, you know?" -> "it was hard."
                    if after.startswith("?"):
                        question_to_period.add(end)
                elif after.startswith(" "):
                    end += 1  # "So like we" -> "So we"
            if cuts and start <= cuts[-1][1]:
                cuts[-1][1] = max(cuts[-1][1], end)
            else:
                cuts.append([start, end])

        pieces, pos = [], 0
        for start, end in cuts:
            pieces.append(text[pos:start])
            pos = end
            if end in question_to_period:
                pieces.append(".")
                pos += 1
        pieces.append(text[pos:])
        cleaned = "".join(pieces)

        cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)  # "word ." -> "word."
        cleaned = re.sub(r",\s*([,.!?;:])", r"\1", cleaned)  # ",." or ",," left behind
        cleaned = re.sub(r"([.!?])\s*,", r"\1", cleaned)  # ". ," left behind
        cleaned = re.sub(r"^[\s,.;:…]+", "", cleaned)  # leading punctuation after "Um,"
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    def highlight_fillers(self, text: str, template: str = ":orange-background[{}]") -> str:
        """Wrap each filler in `template` (Streamlit markdown by default)."""
        pieces, pos = [], 0
        for f in self.find_fillers(text):
            pieces += [text[pos:f.start], template.format(text[f.start:f.end])]
            pos = f.end
        return "".join(pieces) + text[pos:]

    def highlight_html(self, text: str) -> str:
        """Escaped HTML with each filler wrapped in <mark class="vx-filler"> (styled in styles.css)."""
        pieces, pos = [], 0
        for f in self.find_fillers(text):
            pieces += [html.escape(text[pos:f.start]),
                       f'<mark class="vx-filler" title="{html.escape(f.label)}">'
                       f"{html.escape(text[f.start:f.end])}</mark>"]
            pos = f.end
        return "".join(pieces) + html.escape(text[pos:])

    # --------------------------- abbreviations --------------------------- #

    def _replace_chat_abbreviation(self, match: re.Match) -> str:
        token = match.group(0)
        key = token.lower()
        # Upper-case tokens are acronyms, not chat shorthand: "ATM" is not "at the moment".
        if len(token) > 1 and token.isupper():
            return token
        # One/two-letter shorthand only when lower-case: "u" -> "you", but "U" or "Ok" stay.
        if len(key) <= 2 and token != key:
            return token
        expansion = self.abbreviations[key]
        return expansion[0].upper() + expansion[1:] if token[0].isupper() else expansion

    def expand_abbreviations(self, text: str, seen_acronyms: set[str] | None = None) -> str:
        """Expand chat abbreviations, and explain each acronym the first time it appears.

        Pass the same `seen_acronyms` set for consecutive sentences so an acronym
        is only explained once per transcript.
        """
        seen = seen_acronyms if seen_acronyms is not None else set()
        if self._abbr_re:
            text = self._abbr_re.sub(self._replace_chat_abbreviation, text)
        if self._acronym_re:
            def explain(match: re.Match) -> str:
                acronym = match.group(1)
                meaning = self.acronyms[acronym]
                nearby = text[max(0, match.start() - 60):match.end() + 3].lower()
                if acronym in seen or meaning.lower() in nearby or text[match.end():].startswith(" ("):
                    seen.add(acronym)  # already explained in the text itself
                    return acronym
                seen.add(acronym)
                return f"{acronym} ({meaning})"
            text = self._acronym_re.sub(explain, text)
        return text

    # ------------------------------ cleaning ----------------------------- #

    def _clean_sentence(self, text: str, seen_acronyms: set[str]) -> str:
        cleaned = self.expand_abbreviations(self.remove_fillers(text), seen_acronyms)
        # Removing a leading filler ("Um, so...") can leave a lower-case start.
        return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned

    def clean(self, text: str, seen_acronyms: set[str] | None = None) -> str:
        """Clean text sentence by sentence, so each sentence starts with a capital."""
        seen = seen_acronyms if seen_acronyms is not None else set()
        sentences = [self._clean_sentence(s.text, seen) for s in split_sentences(text)]
        return " ".join(s for s in sentences if s.strip(" .,!?"))

    def clean_segments(self, segments: list[Segment]) -> list[Segment]:
        """Clean each segment, keeping timestamps; segments that were only fillers are dropped."""
        seen: set[str] = set()
        cleaned = []
        for s in segments:
            text = self.clean(s.text, seen)
            if text.strip(" .,!?"):
                cleaned.append(Segment(text, s.start_ms, s.end_ms, s.sentiment))
        return cleaned
