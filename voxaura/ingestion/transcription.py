"""Speech-to-text using the AssemblyAI REST API.

Flow: upload audio -> request transcript -> poll until complete -> fetch sentences.
Sentences carry start/end timestamps, which later become the source references
shown next to RAG answers.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

BASE_URL = "https://api.assemblyai.com/v2"


class TranscriptionError(RuntimeError):
    pass


@dataclass
class Segment:
    """A sentence of the transcript. Times are in milliseconds (None if unknown)."""

    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    sentiment: str | None = None  # POSITIVE / NEUTRAL / NEGATIVE (AssemblyAI), if available

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptResult:
    text: str
    segments: list[Segment]
    duration_s: float | None = None


def format_timestamp(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    seconds = int(ms // 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def split_sentences(text: str) -> list[Segment]:
    """Fallback for transcripts without timestamps (e.g. pasted text).

    Splits on sentence-ending punctuation followed by a capital letter or digit,
    and on line breaks (so "Speaker A: ..." lines stay separate). The full stops
    inside "U.S." or "e.g." are therefore not treated as sentence ends.
    """
    parts = re.split(r"(?<=[.!?])(?<![A-Z]\.[A-Z]\.)(?<!\be\.g\.)(?<!\bi\.e\.)\s+(?=[A-Z0-9\"'])|\s*\n\s*",
                     text.strip())
    return [Segment(p.strip()) for p in parts if p.strip()]


class AssemblyAITranscriber:
    def __init__(self, api_key: str, poll_interval: float = 3.0, timeout_s: float = 1800,
                 session: requests.Session | None = None):
        if not api_key:
            raise TranscriptionError("ASSEMBLYAI_API_KEY is not set.")
        self.poll_interval = poll_interval
        self.timeout_s = timeout_s
        self.http = session or requests.Session()
        self.http.headers.update({"authorization": api_key})

    def _upload(self, audio_path: Path) -> str:
        with open(audio_path, "rb") as f:
            response = self.http.post(f"{BASE_URL}/upload", data=f, timeout=300)
        response.raise_for_status()
        return response.json()["upload_url"]

    def _request_transcript(self, audio_url: str) -> str:
        payload = {
            "audio_url": audio_url,
            # Keep "um", "uh" etc. in the raw transcript so we can detect and count
            # them ourselves; AssemblyAI drops them by default.
            "disfluencies": True,
            "punctuate": True,
            "format_text": True,
            # Sentence-level sentiment gives measured evidence for questions about
            # tone, instead of relying only on the LLM's reading of the words.
            "sentiment_analysis": True,
        }
        response = self.http.post(f"{BASE_URL}/transcript", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["id"]

    def _wait_for_completion(self, transcript_id: str) -> dict:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            response = self.http.get(f"{BASE_URL}/transcript/{transcript_id}", timeout=30)
            response.raise_for_status()
            data = response.json()
            if data["status"] == "completed":
                return data
            if data["status"] == "error":
                raise TranscriptionError(f"AssemblyAI error: {data.get('error')}")
            time.sleep(self.poll_interval)
        raise TranscriptionError("Timed out waiting for the transcript.")

    def _sentences(self, transcript_id: str) -> list[Segment]:
        response = self.http.get(f"{BASE_URL}/transcript/{transcript_id}/sentences", timeout=30)
        response.raise_for_status()
        return [
            Segment(s["text"], s.get("start"), s.get("end"))
            for s in response.json().get("sentences", [])
        ]

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        try:
            transcript_id = self._request_transcript(self._upload(audio_path))
            data = self._wait_for_completion(transcript_id)
            text = data.get("text") or ""
            if not text.strip():
                raise TranscriptionError("No speech was detected in the audio.")
            # Sentiment results are already timestamped sentences; use them when present.
            segments = [
                Segment(r["text"], r.get("start"), r.get("end"), r.get("sentiment"))
                for r in data.get("sentiment_analysis_results") or []
            ] or self._sentences(transcript_id) or split_sentences(text)
            return TranscriptResult(text, segments, data.get("audio_duration"))
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 401:
                raise TranscriptionError("AssemblyAI rejected the API key (401). "
                                         "Check ASSEMBLYAI_API_KEY in your .env file.") from e
            if code == 400:
                raise TranscriptionError(f"AssemblyAI could not process this audio: {e.response.text[:200]}") from e
            raise TranscriptionError(f"AssemblyAI request failed: {e}") from e
        except requests.RequestException as e:
            raise TranscriptionError(f"Could not reach AssemblyAI: {e}") from e
