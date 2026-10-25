"""The shared processing pipeline used by every input method.

    audio file -> transcribe -> clean (fillers, abbreviations) -> save
               -> chunk -> embed/index -> AI insights + suggested questions

It has no Streamlit code, so it can be tested and reused on its own.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from pathlib import Path
from typing import Callable

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings
from .database import Database
from .insights import Insights, generate_insights
from .rag import Chunk, TranscriptChat, TranscriptIndex, chunk_segments
from .text_cleaning import FillerStats, TextCleaner
from .transcript_files import read_transcript
from .transcription import AssemblyAITranscriber, Segment, format_timestamp, split_sentences

log = logging.getLogger(__name__)
ProgressCallback = Callable[[str], None]

MIN_WORDS_FOR_INSIGHTS = 5


def build_recording_facts(session: dict) -> str:
    """Measured facts about the recording, given to the LLM as evidence for style/tone questions."""
    segments = session.get("segments") or []
    stats = FillerStats.from_dict(session.get("filler_stats"))
    words = len(session["text"].split())
    facts = [f"- Words (cleaned transcript): {words}"]

    end_ms = segments[-1].get("end_ms") if segments else None
    if end_ms:
        minutes = end_ms / 60_000
        facts.append(f"- Duration: {format_timestamp(end_ms)}")
        if minutes >= 0.25:
            facts.append(f"- Speaking rate: about {round(stats.total_words / minutes)} words per minute "
                         "(typical conversational speech is 120-160)")
    facts.append(f"- Filler words: {stats.total_fillers} ({stats.per_100_words} per 100 words)"
                 + (f"; most common: {', '.join(list(stats.counts)[:4])}" if stats.counts else ""))

    sentiments = Counter(s.get("sentiment") for s in segments if s.get("sentiment"))
    if sentiments:
        total = sum(sentiments.values())
        parts = ", ".join(f"{label.lower()} {round(100 * n / total)}%" for label, n in sentiments.most_common())
        facts.append(f"- Sentence sentiment from AssemblyAI's model ({total} sentences): {parts}")
    else:
        facts.append("- Sentence sentiment: not available for this session")
    facts.append(f"- Input type: {session.get('source_type', 'unknown')}")
    return "\n".join(facts)


class VoxAuraPipeline:
    def __init__(self, settings: Settings, db: Database, cleaner: TextCleaner,
                 embeddings: Embeddings, llm: BaseChatModel,
                 transcriber: AssemblyAITranscriber | None = None):
        self.settings = settings
        self.db = db
        self.cleaner = cleaner
        self.embeddings = embeddings
        self.llm = llm
        self.transcriber = transcriber
        self._chats: dict[str, TranscriptChat] = {}  # one ready-to-use chat per session

    # ------------------------- ingestion ------------------------- #

    def process_audio(self, audio_path: Path, source_type: str, source_name: str,
                      progress: ProgressCallback = lambda _: None) -> dict:
        if self.transcriber is None:
            self.transcriber = AssemblyAITranscriber(self.settings.assemblyai_api_key)
        progress("Uploading and transcribing with AssemblyAI - usually 15-60 seconds...")
        result = self.transcriber.transcribe(audio_path)
        progress(f"Transcribed {len(result.text.split())} words.")
        return self._finish(result.text, result.segments, source_type, source_name, progress)

    def process_text(self, source_name: str, data: bytes | None = None, text: str | None = None,
                     progress: ProgressCallback = lambda _: None) -> dict:
        """Use an existing transcript (uploaded file bytes or pasted text) - skips speech-to-text."""
        progress("Reading and validating the transcript...")
        raw_text, segments = read_transcript(source_name, data=data, text=text)
        return self._finish(raw_text, segments, "text", source_name, progress)

    def _finish(self, raw_text: str, raw_segments: list[Segment], source_type: str,
                source_name: str, progress: ProgressCallback) -> dict:
        progress("Detecting filler words and expanding abbreviations...")
        filler_stats = self.cleaner.detect_fillers(raw_text)
        segments = self.cleaner.clean_segments(raw_segments)
        if not segments:
            raise ValueError("Nothing was left after removing filler words - the recording may be empty.")
        session = {
            "session_id": uuid.uuid4().hex[:16],
            "title": source_name[:120],
            "source_type": source_type,
            "source_name": source_name,
            "text": " ".join(s.text for s in segments),
            "raw_text": raw_text,
            "segments": [s.to_dict() for s in segments],
            "filler_stats": filler_stats.to_dict(),
            "insights": None,
            "warnings": [],  # shown once in the UI, not saved
        }
        progress("Saving the session...")
        self.db.save_session(session)
        progress("Chunking and embedding the transcript into the vector database...")
        self.build_index(session)

        if len(session["text"].split()) >= MIN_WORDS_FOR_INSIGHTS:
            progress("Generating AI insights and suggested questions...")
            try:
                self.create_insights(session)
            except Exception as e:  # the session is still usable without insights
                log.exception("Insight generation failed")
                session["warnings"].append(f"AI insights could not be generated ({e}). "
                                           "You can retry from the AI insights page.")
        return session

    # ----------------------------- RAG ----------------------------- #

    def chunks_for(self, session: dict) -> list[Chunk]:
        segments = (
            [Segment(**s) for s in session["segments"]]
            if session.get("segments")
            else split_sentences(session["text"])  # sessions saved by the first version
        )
        return chunk_segments(segments, self.settings.chunk_size, self.settings.chunk_overlap)

    def index_for(self, session: dict) -> TranscriptIndex:
        return TranscriptIndex(session["session_id"], self.embeddings, self.settings.vectorstore_dir)

    def build_index(self, session: dict) -> TranscriptIndex:
        index = self.index_for(session)
        index.build(self.chunks_for(session))
        return index

    def chat(self, session: dict) -> TranscriptChat:
        sid = session["session_id"]
        if sid not in self._chats:
            self._chats[sid] = TranscriptChat(
                self.llm, self.build_index(session), k=self.settings.retrieval_k,
                full_context_chars=self.settings.full_context_chars,
                recording_facts=build_recording_facts(session),
            )
        return self._chats[sid]

    # --------------------------- insights --------------------------- #

    def create_insights(self, session: dict) -> Insights:
        insights = generate_insights(self.llm, session["text"], self.chunks_for(session))
        self.db.update_insights(session["session_id"], insights.to_dict(), title=insights.title or None)
        session["insights"] = insights.to_dict()
        if insights.title:
            session["title"] = insights.title
        return insights

    def delete_session(self, session: dict) -> None:
        self.db.delete_session(session["session_id"])
        self._chats.pop(session["session_id"], None)
        try:
            self.index_for(session).delete()
        except Exception:
            pass  # the vector index may never have been built
