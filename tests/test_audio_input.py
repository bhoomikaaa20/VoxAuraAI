import array
import math
import wave

import pytest

from voxaura.audio_input import AudioInputError, check_recording, save_audio_bytes


def write_wav(path, seconds, amplitude, rate=16000):
    samples = array.array("h", (int(amplitude * math.sin(i / 8)) for i in range(int(seconds * rate))))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return path


def test_valid_recording_passes(tmp_path):
    assert check_recording(write_wav(tmp_path / "ok.wav", 2.0, 8000)) == pytest.approx(2.0)


def test_too_short_recording_is_rejected(tmp_path):
    with pytest.raises(AudioInputError, match="only 0.3 s"):
        check_recording(write_wav(tmp_path / "short.wav", 0.3, 8000))


def test_silent_recording_is_rejected(tmp_path):
    with pytest.raises(AudioInputError, match="silent"):
        check_recording(write_wav(tmp_path / "silent.wav", 2.0, 0))


def test_corrupt_recording_is_rejected(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav file at all")
    with pytest.raises(AudioInputError, match="could not be read"):
        check_recording(bad)


def test_save_audio_bytes_validates_type_and_content(tmp_path):
    assert save_audio_bytes(b"data", "talk.MP3", tmp_path).suffix == ".mp3"
    with pytest.raises(AudioInputError):
        save_audio_bytes(b"", "talk.mp3", tmp_path)
    with pytest.raises(AudioInputError):
        save_audio_bytes(b"data", "virus.exe", tmp_path)
