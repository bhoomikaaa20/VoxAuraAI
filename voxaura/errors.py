"""Turn exceptions into short messages that tell the user what to do next."""

from __future__ import annotations

import requests

from .database import DatabaseError
from .ingestion.audio_input import AudioInputError
from .ingestion.transcript_files import TranscriptFileError
from .ingestion.transcription import TranscriptionError


def friendly_error(e: Exception) -> str:
    if isinstance(e, (AudioInputError, TranscriptionError, TranscriptFileError, DatabaseError, ValueError)):
        return str(e)
    name = type(e).__name__
    if name == "AuthenticationError":
        return "OpenAI rejected the API key. Put a valid OPENAI_API_KEY in your .env file, then refresh the page."
    if name == "RateLimitError":
        return "OpenAI rate limit or quota reached. Wait a moment, or check your OpenAI billing."
    if name in ("APIConnectionError", "APITimeoutError") or isinstance(e, requests.ConnectionError):
        return "Could not reach the AI service. Check your internet connection and try again."
    return f"Unexpected error ({name}): {str(e)[:200]}"
