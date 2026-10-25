"""Turns each supported input (upload, recording, YouTube link) into a local audio file.

Every input method ends in the same place - a file path - so the rest of the
pipeline does not need to know where the audio came from.
"""

from __future__ import annotations

import uuid
from pathlib import Path

SUPPORTED_AUDIO_TYPES = ["mp3", "wav", "m4a", "ogg", "flac", "webm", "mp4"]
MAX_UPLOAD_MB = 200


class AudioInputError(RuntimeError):
    pass


def save_audio_bytes(data: bytes, original_name: str, audio_dir: Path) -> Path:
    """Save uploaded or recorded audio bytes under a unique name."""
    if not data:
        raise AudioInputError("The audio file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise AudioInputError(f"Audio files must be under {MAX_UPLOAD_MB} MB.")
    suffix = Path(original_name).suffix.lower() or ".wav"
    if suffix.lstrip(".") not in SUPPORTED_AUDIO_TYPES:
        raise AudioInputError(f"Unsupported file type '{suffix}'.")
    audio_dir.mkdir(parents=True, exist_ok=True)
    path = audio_dir / f"{uuid.uuid4().hex[:12]}{suffix}"
    path.write_bytes(data)
    return path


def check_recording(path: Path, min_seconds: float = 1.0, silence_peak: int = 300) -> float:
    """Validate a microphone recording (WAV). Returns its duration in seconds.

    Catches the two common failures before paying for transcription: a recording
    that is too short, and a silent one (microphone blocked or wrong input device).
    """
    import array
    import wave

    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            duration = wav.getnframes() / float(wav.getframerate() or 1)
            sample_width = wav.getsampwidth()
    except (wave.Error, EOFError) as e:
        raise AudioInputError(f"The recording could not be read as audio ({e}). Please record again.") from e

    if duration < min_seconds:
        raise AudioInputError(f"The recording is only {duration:.1f} s long. Please record at least "
                              f"{min_seconds:.0f} second(s) of speech.")
    if sample_width == 2 and frames:
        samples = array.array("h", frames[: len(frames) // 2 * 2])
        if max(abs(min(samples)), max(samples)) < silence_peak:
            raise AudioInputError("The recording is silent. Check that the browser has microphone "
                                  "permission and that the right input device is selected.")
    return duration


def download_youtube_audio(url: str, audio_dir: Path) -> tuple[Path, str]:
    """Download the audio track of a YouTube (or other yt-dlp supported) video.

    Returns (file path, video title). The native audio format is kept, so FFmpeg
    is not needed - AssemblyAI accepts m4a/webm directly.
    """
    import yt_dlp  # imported lazily: only needed for this input method

    if not url.startswith(("http://", "https://")):
        raise AudioInputError("Please enter a full URL starting with http:// or https://")
    audio_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(audio_dir / f"{uuid.uuid4().hex[:12]}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
    }
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as e:  # yt-dlp raises many different error types
        raise AudioInputError(f"Could not download audio: {e}") from e
    if not path.exists():
        raise AudioInputError("Download finished but the audio file was not found.")
    return path, info.get("title", url)
