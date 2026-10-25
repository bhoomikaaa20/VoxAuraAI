"""Transcript cleaning: filler-word detection/removal and abbreviation expansion.

The word lists live in config/fillers.json and config/abbreviations.json so they
can be changed without touching code.

Fillers come in two kinds:
  * hesitation sounds ("um", "uh") - always fillers;
  * contextual fillers ("like", "you know", "I mean") - only fillers when they
    are set off by commas, e.g. "It was, like, huge" but not "I like pizza".

Abbreviations also come in two kinds:
  * chat abbreviations ("idk", "btw") are replaced by their meaning;
  * acronyms ("AI", "U.S.") are kept and explained on first use:
    "AI (artificial intelligence)".
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .transcription import Segment

# A token must not be glued to letters, digits, dots, apostrophes, hyphens or "&"
# on either side. This keeps "um" out of "umbrella", "uh" out of "uh-huh", and
# "U" / "US" out of "U.S." or "R&D".
_BEFORE = r"(?<![\w.'&-])"
_AFTER = r"(?![\w'&-]|\.\w)"


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


class TextCleaner:
    def __init__(self, fillers: list[str], abbreviations: dict[str, str],
                 contextual_fillers: list[str] | None = None, acronyms: dict[str, str] | None = None):
        self.fillers = [f.lower() for f in fillers]
        self.contextual_fillers = [f.lower() for f in contextual_fillers or []]
        self.abbreviations = {k.lower(): v for k, v in abbreviations.items()}
        self.acronyms = dict(acronyms or {})

        # Hesitation sounds, plus a trailing comma: "Um, I think" -> "I think".
        self._filler_re = (
            re.compile(_BEFORE + "(" + _alternation(self.fillers) + ")" + _AFTER + ",?", re.IGNORECASE)
            if self.fillers else None
        )
        # Contextual fillers must start the text/sentence or follow a comma, and be
        # followed by a comma. The surrounding commas are removed with them:
        # "It was, like, huge" -> "It was huge".
        self._contextual_re = (
            re.compile(r"(?:^|(?<=[.!?] )|, )(" + _alternation(self.contextual_fillers) + r"),",
                       re.IGNORECASE | re.MULTILINE)
            if self.contextual_fillers else None
        )
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
        return cls(
            fillers=fillers["fillers"],
            contextual_fillers=fillers.get("contextual_fillers", []),
            abbreviations=abbreviations["chat_abbreviations"],
            acronyms=abbreviations.get("acronyms", {}),
        )

    # ------------------------------ fillers ------------------------------ #

    def detect_fillers(self, text: str) -> FillerStats:
        counts: Counter[str] = Counter()
        for pattern in (self._filler_re, self._contextual_re):
            if pattern:
                for match in pattern.finditer(text):
                    counts[match.group(1).lower()] += 1
        return FillerStats(counts=dict(counts.most_common()), total_words=len(text.split()))

    def remove_fillers(self, text: str) -> str:
        if self._contextual_re:
            text = self._contextual_re.sub("", text)
        if self._filler_re:
            text = self._filler_re.sub("", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)  # "word ." -> "word."
        text = re.sub(r"([.!?,])\s*,", r"\1", text)  # ". ," or ", ," left behind
        text = re.sub(r"^[\s,]+", "", text)  # leading comma after removing "Um,"
        return re.sub(r"\s{2,}", " ", text).strip()

    def highlight_fillers(self, text: str) -> str:
        """Return Streamlit markdown with filler words highlighted."""
        for pattern in (self._contextual_re, self._filler_re):
            if pattern:
                text = pattern.sub(lambda m: m.group(0).replace(
                    m.group(1), f":orange-background[{m.group(1)}]"), text)
        return text

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

    def clean(self, text: str, seen_acronyms: set[str] | None = None) -> str:
        cleaned = self.expand_abbreviations(self.remove_fillers(text), seen_acronyms)
        # Removing a leading filler ("Um, so...") can leave a lower-case start.
        return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned

    def clean_segments(self, segments: list[Segment]) -> list[Segment]:
        """Clean each sentence, keeping timestamps; sentences that were only fillers are dropped."""
        seen: set[str] = set()
        cleaned = []
        for s in segments:
            text = self.clean(s.text, seen)
            if text.strip(" .,!?"):
                cleaned.append(Segment(text, s.start_ms, s.end_ms, s.sentiment))
        return cleaned
