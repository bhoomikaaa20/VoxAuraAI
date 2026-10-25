"""Shared test fixtures. No API keys or network access are needed."""

import hashlib
import logging
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxaura.config import PROJECT_ROOT, load_settings  # noqa: E402
from voxaura.analysis.text_cleaning import TextCleaner  # noqa: E402


class BagOfWordsEmbeddings(Embeddings):
    """A tiny deterministic embedding: hashed word counts, L2-normalised.

    Texts that share words get similar vectors, which is enough to test that
    retrieval returns the relevant chunk - without calling OpenAI.
    """

    dims = 256
    stopwords = {"the", "a", "an", "is", "was", "of", "to", "and", "in", "what", "did", "we", "about"}

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z]+", text.lower()):
            if word not in self.stopwords:
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class RecordingChatModel(FakeListChatModel):
    """Returns canned responses in order and records every prompt it receives."""

    calls: list = Field(default_factory=list)

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        return super()._call(messages, stop, run_manager, **kwargs)


INSIGHTS_JSON = ('{"title": "Phone launch", "summary": "A product launch.", "key_points": ["Battery"], '
                 '"action_items": ["Open pre-orders on Friday"], "tone": "Upbeat and promotional.", '
                 '"suggested_questions": ["How long does the battery last?", "Why is the speaker excited?"]}')


class ScriptedChatModel(FakeListChatModel):
    """Replies according to which prompt it receives, like a well-behaved real model."""

    responses: list = Field(default_factory=lambda: ["unused"])
    calls: list = Field(default_factory=list)

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        system = messages[0].content
        if "return a JSON object" in system:
            return INSIGHTS_JSON
        if "Rewrite the user's latest question" in system:
            return messages[-1].content.replace("Latest question: ", "")
        return f"Answer to: {messages[-1].content} [1]"


@pytest.fixture(autouse=True)
def quiet_chroma():
    logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


@pytest.fixture
def embeddings():
    return BagOfWordsEmbeddings()


@pytest.fixture
def cleaner():
    return TextCleaner.from_config_dir(PROJECT_ROOT / "config")


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_HOST", "")  # force SQLite even if .env configures MySQL
    return replace(
        load_settings(),
        db_host="",
        sqlite_path=tmp_path / "test.db",
        vectorstore_dir=tmp_path / "vectorstore",
        audio_dir=tmp_path / "audio",
        chunk_size=200,
        chunk_overlap=40,
        retrieval_k=2,
    )
