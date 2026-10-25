"""End-to-end tests of the shared pipeline with fake speech-to-text and LLM."""

import pytest

from voxaura.database import Database
from voxaura.pipeline import VoxAuraPipeline, build_recording_facts
from voxaura.transcript_files import TranscriptFileError
from voxaura.transcription import AssemblyAITranscriber, Segment, TranscriptionError, TranscriptResult
from tests.conftest import RecordingChatModel, ScriptedChatModel
from tests.test_audio_input import write_wav


class FakeTranscriber:
    def transcribe(self, audio_path):
        segments = [
            Segment("Um, welcome to the product launch.", 0, 3000, "POSITIVE"),
            Segment("The new phone has a uh three day battery life.", 3000, 7000, "POSITIVE"),
            Segment("Pre-orders open on Friday.", 7000, 9000, "NEUTRAL"),
        ]
        return TranscriptResult(" ".join(s.text for s in segments), segments, 9.0)


def make_pipeline(settings, cleaner, embeddings, llm):
    return VoxAuraPipeline(settings, Database.from_settings(settings), cleaner, embeddings, llm,
                           transcriber=FakeTranscriber())


def test_audio_to_insights_and_grounded_answer(settings, cleaner, embeddings, tmp_path):
    llm = ScriptedChatModel()
    pipeline = make_pipeline(settings, cleaner, embeddings, llm)
    progress = []

    session = pipeline.process_audio(tmp_path / "launch.mp3", "upload", "launch.mp3", progress.append)

    # cleaning + filler detection, timestamps and sentiment kept
    assert session["text"].startswith("Welcome to the product launch.")
    assert "uh" not in session["text"].split()
    assert session["filler_stats"]["counts"] == {"um": 1, "uh": 1}
    assert session["segments"][1] == {"text": "The new phone has a three day battery life.",
                                      "start_ms": 3000, "end_ms": 7000, "sentiment": "POSITIVE"}
    assert any("insights" in p for p in progress)

    # insights + suggestions are generated automatically and saved
    assert session["title"] == "Phone launch"
    assert session["insights"]["suggested_questions"][0] == "How long does the battery last?"
    saved = pipeline.db.get_session(session["session_id"])
    assert saved["raw_text"].startswith("Um,") and saved["insights"]["tone"] == "Upbeat and promotional."

    # RAG over the indexed transcript, with measured facts in the prompt
    answer = pipeline.chat(session).ask("How long is the battery life?")
    assert "battery" in answer.sources[0].chunk.text
    assert "positive 67%" in llm.calls[-1][0].content
    assert pipeline.chat(session) is pipeline.chat(session)  # reused, not rebuilt per question


def test_insight_failure_does_not_lose_the_session(settings, cleaner, embeddings, tmp_path):
    llm = RecordingChatModel(responses=["this is not JSON"])
    pipeline = make_pipeline(settings, cleaner, embeddings, llm)
    session = pipeline.process_audio(tmp_path / "a.mp3", "upload", "a.mp3")
    assert session["insights"] is None
    assert "could not be generated" in session["warnings"][0]
    assert pipeline.db.get_session(session["session_id"]) is not None


def test_recording_flow(settings, cleaner, embeddings, tmp_path):
    from voxaura.audio_input import check_recording

    path = write_wav(tmp_path / "rec.wav", 2.0, 8000)
    assert check_recording(path) == pytest.approx(2.0)
    session = make_pipeline(settings, cleaner, embeddings, ScriptedChatModel()).process_audio(
        path, "recording", "Recording 1")
    assert session["source_type"] == "recording" and session["insights"] is not None


def test_text_upload_flow(settings, cleaner, embeddings):
    pipeline = make_pipeline(settings, cleaner, embeddings, ScriptedChatModel())
    data = "We launch on Friday. Tell the team asap, you know, before the AI demo.".encode("utf-16")
    session = pipeline.process_text("notes.txt", data=data)

    assert session["source_type"] == "text"
    assert "as soon as possible" in session["text"]
    assert "AI (artificial intelligence)" in session["text"]
    assert session["filler_stats"]["counts"] == {"you know": 1}
    assert pipeline.chat(session).ask("When is the launch?").sources

    pipeline.delete_session(session)
    assert pipeline.db.get_session(session["session_id"]) is None


def test_invalid_text_upload_is_rejected_before_saving(settings, cleaner, embeddings):
    pipeline = make_pipeline(settings, cleaner, embeddings, ScriptedChatModel())
    with pytest.raises(TranscriptFileError):
        pipeline.process_text("tiny.txt", data=b"hi")
    assert pipeline.db.list_sessions() == []


def test_recording_facts():
    session = {"text": "word " * 150, "source_type": "recording",
               "segments": [{"text": "a", "start_ms": 0, "end_ms": 60_000, "sentiment": "NEGATIVE"}],
               "filler_stats": {"counts": {"um": 3}, "total_words": 150}}
    facts = build_recording_facts(session)
    assert "about 150 words per minute" in facts
    assert "3 (2.0 per 100 words); most common: um" in facts
    assert "negative 100%" in facts


class FakeResponse:
    def __init__(self, data, status=200):
        self.data = data
        self.status_code = status
        self.text = str(data)

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self.data


class FakeHttp:
    """Mimics the AssemblyAI REST API: upload, create, poll (processing -> completed)."""

    def __init__(self, upload_status=200):
        self.headers = {}
        self.polls = 0
        self.payload = None
        self.upload_status = upload_status

    def post(self, url, json=None, data=None, timeout=None):
        if url.endswith("/upload"):
            return FakeResponse({"upload_url": "https://cdn/audio"}, self.upload_status)
        self.payload = json
        return FakeResponse({"id": "t1"})

    def get(self, url, timeout=None):
        if url.endswith("/sentences"):
            return FakeResponse({"sentences": [{"text": "Hello there.", "start": 0, "end": 800}]})
        self.polls += 1
        status = "completed" if self.polls >= 2 else "processing"
        return FakeResponse({"status": status, "text": "Hello there.", "audio_duration": 1,
                             "sentiment_analysis_results": [
                                 {"text": "Hello there.", "start": 0, "end": 800, "sentiment": "POSITIVE"}]})


def test_assemblyai_client_polls_until_complete(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    http = FakeHttp()
    result = AssemblyAITranscriber("key", poll_interval=0, session=http).transcribe(audio)

    assert http.polls == 2
    assert http.payload["disfluencies"] is True  # keeps fillers so we can detect them
    assert http.payload["sentiment_analysis"] is True
    assert result.segments == [Segment("Hello there.", 0, 800, "POSITIVE")]


def test_assemblyai_bad_key_gives_clear_error(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF")
    with pytest.raises(TranscriptionError, match="rejected the API key"):
        AssemblyAITranscriber("bad", poll_interval=0, session=FakeHttp(upload_status=401)).transcribe(audio)
