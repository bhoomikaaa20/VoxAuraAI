import re

import pytest

from voxaura.rag import chunk_segments
from voxaura.transcription import Segment, split_sentences


def make_segments(n, length=50):
    return [Segment(f"Sentence {i:02d} " + "x" * (length - 12) + ".", i * 1000, i * 1000 + 900)
            for i in range(n)]


def test_chunks_respect_size_and_never_split_sentences():
    segments = make_segments(20)
    chunks = chunk_segments(segments, chunk_size=200, chunk_overlap=0)
    assert len(chunks) > 1
    whole_sentences = re.compile(r"Sentence \d\d x{38}\.( Sentence \d\d x{38}\.)*")
    for chunk in chunks:
        assert len(chunk.text) <= 200
        assert whole_sentences.fullmatch(chunk.text)


def test_overlap_repeats_trailing_sentence():
    chunks = chunk_segments(make_segments(10), chunk_size=200, chunk_overlap=60)
    last_sentence_of_first = chunks[0].text.split(". ")[-1]
    assert chunks[1].text.startswith(last_sentence_of_first.rstrip("."))


def test_chunks_keep_time_ranges():
    chunks = chunk_segments(make_segments(10), chunk_size=200, chunk_overlap=0)
    assert chunks[0].start_ms == 0
    assert chunks[-1].end_ms == 9 * 1000 + 900
    assert chunks[0].time_range == "00:00-00:02"
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_very_long_sentence_is_split_on_words():
    chunks = chunk_segments([Segment("word " * 100)], chunk_size=100, chunk_overlap=0)
    assert len(chunks) >= 5
    assert all(len(c.text) <= 100 for c in chunks)


def test_plain_text_fallback_has_no_timestamps():
    segments = split_sentences("First point. Second point! Third point?")
    assert [s.text for s in segments] == ["First point.", "Second point!", "Third point?"]
    assert chunk_segments(segments)[0].time_range == "no timestamp"


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_segments(make_segments(3), chunk_size=100, chunk_overlap=100)
