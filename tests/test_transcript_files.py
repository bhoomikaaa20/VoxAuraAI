import pytest

from voxaura.transcript_files import TranscriptFileError, decode_text, read_transcript
from voxaura.transcription import split_sentences

SRT = """1
00:00:01,000 --> 00:00:04,500
Welcome to the show.

2
00:00:05,000 --> 00:01:02,250
Today we talk about <i>climate</i> policy.
"""

VTT = """WEBVTT

00:01.000 --> 00:03.000
<v Anna>Hello and welcome everyone.

00:03.500 --> 00:06.000
<v Ben>Thanks for having me today.
"""


def test_plain_text_upload():
    text, segments = read_transcript("notes.txt", data="First point here. Second point now.".encode())
    assert text == "First point here. Second point now."
    assert [s.text for s in segments] == ["First point here.", "Second point now."]
    assert segments[0].start_ms is None


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "cp1252"])
def test_common_encodings_are_decoded(encoding):
    original = "Café meeting notes – we agreed on the plan."
    assert decode_text(original.encode(encoding)) == original


def test_srt_keeps_timestamps_and_strips_tags():
    text, segments = read_transcript("show.srt", data=SRT.encode())
    assert [s.text for s in segments] == ["Welcome to the show.", "Today we talk about climate policy."]
    assert (segments[1].start_ms, segments[1].end_ms) == (5000, 62250)
    assert text.startswith("Welcome to the show.")


def test_vtt_with_speaker_tags():
    _, segments = read_transcript("talk.vtt", data=VTT.encode())
    assert segments[0].text == "Hello and welcome everyone."
    assert segments[1].start_ms == 3500


def test_pasted_text_and_line_breaks():
    _, segments = read_transcript("Pasted transcript", text="Speaker A: hi there everyone\nSpeaker B: hello to you all")
    assert [s.text for s in segments] == ["Speaker A: hi there everyone", "Speaker B: hello to you all"]


@pytest.mark.parametrize("name, data, message", [
    ("a.txt", b"too short", "too short"),
    ("a.txt", b"\x00\x01\x02binary" * 10, "binary"),
    ("a.srt", b"1\nno timings here at all, sorry friend\n", "No subtitle cues"),
])
def test_invalid_transcripts_are_rejected(name, data, message):
    with pytest.raises(TranscriptFileError, match=message):
        read_transcript(name, data=data)


def test_sentence_split_does_not_break_us():
    parts = [s.text for s in split_sentences("The U.S. Economy grew fast. Then it slowed.")]
    assert parts == ["The U.S. Economy grew fast.", "Then it slowed."]
