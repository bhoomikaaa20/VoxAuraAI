import json

import pytest

from voxaura.text_cleaning import TextCleaner
from voxaura.transcription import Segment


# ------------------------------ fillers ------------------------------ #

def test_removes_hesitation_fillers_and_tidies_punctuation(cleaner):
    assert cleaner.clean("Um, I think uh we should start.") == "I think we should start."


def test_does_not_touch_words_containing_fillers(cleaner):
    # "um" is inside "umbrella", "err" inside "error", "uh" inside "uh-huh"
    assert cleaner.clean("The umbrella error was fixed, uh-huh.") == "The umbrella error was fixed, uh-huh."


@pytest.mark.parametrize("text, expected", [
    ("It was, like, huge.", "It was huge."),
    ("You know, we tried everything.", "We tried everything."),
    ("So, I mean, the results were good.", "So the results were good."),
    ("It was, you know what I mean, strange.", "It was strange."),
])
def test_removes_contextual_fillers_set_off_by_commas(cleaner, text, expected):
    assert cleaner.clean(text) == expected


@pytest.mark.parametrize("text", [
    "I like pizza.",
    "Do you know the answer?",
    "I mean what I say.",
    "This kind of car is rare.",
])
def test_keeps_the_same_words_when_they_carry_meaning(cleaner, text):
    assert cleaner.remove_fillers(text) == text


def test_detects_and_counts_fillers_including_phrases(cleaner):
    stats = cleaner.detect_fillers("Um so, UM, I uh think, you know, it was, like, umm fine. I like it.")
    assert stats.counts == {"um": 2, "uh": 1, "umm": 1, "you know": 1, "like": 1}
    assert stats.total_fillers == 6


def test_us_and_acronyms_are_never_fillers(cleaner):
    stats = cleaner.detect_fillers("The U.S. and the UK told us. Uh, the US.")
    assert stats.counts == {"uh": 1}


def test_highlight_marks_fillers_only(cleaner):
    assert cleaner.highlight_fillers("So um yes") == "So :orange-background[um] yes"
    assert ":orange-background[like]" in cleaner.highlight_fillers("It was, like, big")
    assert cleaner.highlight_fillers("I like it") == "I like it"


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
