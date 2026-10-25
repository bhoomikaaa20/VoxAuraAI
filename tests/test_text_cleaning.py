import json

import pytest

from voxaura.analysis.text_cleaning import TextCleaner
from voxaura.ingestion.transcription import Segment


# ------------------------------ fillers ------------------------------ #

def test_removes_hesitation_fillers_and_tidies_punctuation(cleaner):
    assert cleaner.clean("Um, I think uh we should start.") == "I think we should start."


def test_does_not_touch_words_containing_fillers(cleaner):
    # "um" is inside "umbrella", "err" inside "error", "uh" inside "uh-huh"
    assert cleaner.clean("The umbrella error was fixed, uh-huh.") == "The umbrella error was fixed, uh-huh."


@pytest.mark.parametrize("text, expected", [
    # set off by commas
    ("It was, like, huge.", "It was huge."),
    ("You know, we tried everything.", "We tried everything."),
    ("So, I mean, the results were good.", "So the results were good."),
    ("It was, you know what I mean, strange.", "It was strange."),
    # no commas - how people (and speech-to-text) usually write them
    ("So like I was like really tired.", "So I was really tired."),
    ("And you know I just felt bad.", "And I just felt bad."),
    ("I mean it was okay I guess.", "It was okay."),
    ("Basically we need more time.", "We need more time."),
    ("There were like 20 people there.", "There were 20 people there."),
    ("It was kind of good, right?", "It was good."),
    ("It was hard, you know?", "It was hard."),
    ("We need snacks or something.", "We need snacks."),
])
def test_removes_discourse_marker_fillers(cleaner, text, expected):
    assert cleaner.clean(text) == expected


@pytest.mark.parametrize("text", [
    "I like pizza.",
    "It looks like rain.",
    "It's like a dream.",
    "Something like that.",
    "Do you know the answer?",
    "You know what happened next?",
    "I mean what I say.",
    "I mean it.",
    "This kind of car is rare.",
    "Well done!",
    "We did it as well.",
    "I actually like it.",
    "The crime will soon, and I mean very soon, come to an end.",
])
def test_keeps_the_same_words_when_they_carry_meaning(cleaner, text):
    assert cleaner.detect_fillers(text).counts == {}
    assert cleaner.remove_fillers(text) == text


def test_real_assemblyai_output_is_cleaned(cleaner):
    """Text returned by AssemblyAI (disfluencies on) for a test recording full of fillers."""
    raw = ("Um, so, uh, I think the project is, like, really good. You know, we worked hard on it. "
           "Uh, I mean, the the results were, um, better than expected. So like we should, uh, present it next week.")
    assert cleaner.detect_fillers(raw).counts == {
        "uh": 3, "um": 2, "like": 2, "you know": 1, "i mean": 1, "repeated word": 1}
    assert cleaner.clean(raw) == ("So I think the project is really good. We worked hard on it. "
                                  "The results were better than expected. So we should present it next week.")


def test_repeated_words(cleaner):
    assert cleaner.clean("I I think the the plan works.") == "I think the plan works."
    assert cleaner.clean("I know that that is true. Bye bye.") == "I know that that is true. Bye bye."


def test_detects_and_counts_fillers_including_phrases(cleaner):
    stats = cleaner.detect_fillers("Um so, UM, I uh think, you know, it was, like, umm fine. I like it.")
    assert stats.counts == {"um": 2, "uh": 1, "umm": 1, "you know": 1, "like": 1}
    assert stats.total_fillers == 6


def test_unknown_rule_in_config_is_reported():
    with pytest.raises(ValueError, match="Unknown filler rule"):
        TextCleaner([], {}, discourse_markers={"like": "sometimes"})


def test_us_and_acronyms_are_never_fillers(cleaner):
    stats = cleaner.detect_fillers("The U.S. and the UK told us. Uh, the US.")
    assert stats.counts == {"uh": 1}


def test_highlight_marks_fillers_only(cleaner):
    assert cleaner.highlight_fillers("So um yes") == "So :orange-background[um] yes"
    assert ":orange-background[like]" in cleaner.highlight_fillers("It was, like, big")
    assert cleaner.highlight_fillers("I like it") == "I like it"
    assert cleaner.highlight_html("<b> um") == '&lt;b&gt; <mark class="vx-filler" title="um">um</mark>'


# --------------------------- abbreviations --------------------------- #

def test_expands_chat_abbreviations_preserving_capitals():
    cleaner = TextCleaner(fillers=[], abbreviations={"btw": "by the way", "idk": "I don't know"})
    assert cleaner.clean("Btw idk.") == "By the way I don't know."


def test_short_and_upper_case_shorthand_is_left_alone(cleaner):
    # config maps "u" -> "you", "k" -> "okay", "atm" -> "at the moment"
    assert cleaner.clean("See u at the ATM, it needs Vitamin K.") == "See you at the ATM, it needs Vitamin K."


def test_acronyms_are_explained_once(cleaner):
    text = cleaner.clean("AI is everywhere. AI helps the NHS.")
    assert text == "AI (artificial intelligence) is everywhere. AI helps the NHS (National Health Service)."


def test_dotted_acronyms_and_pronoun_us(cleaner):
    assert cleaner.clean("The U.S. economy grew. Tell us more.") == \
        "The U.S. (United States) economy grew. Tell us more."
    assert cleaner.clean("Spending on R&D rose.") == "Spending on R&D (research and development) rose."


def test_acronym_already_explained_is_not_repeated(cleaner):
    assert cleaner.clean("Artificial intelligence (AI) is new.") == "Artificial intelligence (AI) is new."


def test_clean_segments_shares_acronym_memory_and_drops_filler_only_sentences(cleaner):
    segments = [Segment("Um.", 0, 500), Segment("We use AI.", 500, 1500), Segment("AI is great.", 1500, 2500)]
    cleaned = cleaner.clean_segments(segments)
    assert [s.text for s in cleaned] == ["We use AI (artificial intelligence).", "AI is great."]
    assert cleaned[0].start_ms == 500


def test_older_flat_abbreviation_config_still_loads(tmp_path):
    (tmp_path / "fillers.json").write_text(json.dumps({"fillers": ["um"]}))
    (tmp_path / "abbreviations.json").write_text(json.dumps({"asap": "as soon as possible"}))
    cleaner = TextCleaner.from_config_dir(tmp_path)
    assert cleaner.clean("Um, do it asap.") == "Do it as soon as possible."


def test_empty_config_leaves_text_unchanged():
    assert TextCleaner([], {}).clean("hello um world") == "Hello um world"
