"""Reading and validating transcripts supplied as text instead of audio.

Supports plain text / Markdown, and SubRip (.srt) / WebVTT (.vtt) subtitle
files. Subtitle cues keep their timestamps, so uploaded subtitles get the same
timestamped source references as transcribed audio.
"""

from __future__ import annotations

import re

from .transcription import Segment, split_sentences

SUPPORTED_TEXT_TYPES = ["txt", "md", "srt", "vtt"]
MAX_TEXT_BYTES = 2 * 1024 * 1024  # 2 MB is roughly 300k words
MIN_WORDS = 5

_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})"
_CUE_TIME_RE = re.compile(_TIME + r"\s*-->\s*" + _TIME)


class TranscriptFileError(ValueError):
    pass


def decode_text(data: bytes) -> str:
    """Decode bytes, handling UTF-8 (with or without BOM), UTF-16 (Windows Notepad) and Latin-1."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data[:1000]:
        raise TranscriptFileError("This looks like a binary file, not a text transcript.")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _to_ms(hours, minutes, seconds, millis) -> int:
    return ((int(hours or 0) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int(millis)


def parse_subtitles(text: str) -> list[Segment]:
    """Parse SRT or VTT cues into timestamped segments."""
    segments = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [ln.strip() for ln in block.strip().split("\n") if ln.strip()]
        for i, line in enumerate(lines):
            match = _CUE_TIME_RE.search(line)
            if match:
                cue = " ".join(lines[i + 1:])
                cue = re.sub(r"<[^>]+>", "", cue).strip()  # VTT styling tags such as <v Speaker>
                if cue:
                    segments.append(Segment(cue, _to_ms(*match.groups()[:4]), _to_ms(*match.groups()[4:])))
                break
    return segments


def normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def read_transcript(name: str, data: bytes | None = None, text: str | None = None) -> tuple[str, list[Segment]]:
    """Validate an uploaded file (name + bytes) or pasted text and return (text, segments)."""
    if data is not None:
        if len(data) > MAX_TEXT_BYTES:
            raise TranscriptFileError("Transcript files must be under 2 MB.")
        text = decode_text(data)
    text = normalise_text(text or "")

    suffix = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix in ("srt", "vtt") or _CUE_TIME_RE.search(text[:500]):
        segments = parse_subtitles(text)
        if not segments:
            raise TranscriptFileError("No subtitle cues were found in this file.")
        text = " ".join(s.text for s in segments)
    else:
        segments = split_sentences(text)

    if len(text.split()) < MIN_WORDS:
        raise TranscriptFileError(f"The transcript is too short - at least {MIN_WORDS} words are needed.")
    return text, segments
