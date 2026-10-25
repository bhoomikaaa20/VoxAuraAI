"""Application settings, read once from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return int(value) if value else default
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}")


@dataclass(frozen=True)
class Settings:
    assemblyai_api_key: str
    openai_api_key: str
    chat_model: str
    embedding_model: str

    # RAG parameters
    chunk_size: int  # target characters per chunk
    chunk_overlap: int  # characters of trailing context repeated in the next chunk
    retrieval_k: int  # number of chunks retrieved per question
    full_context_chars: int  # transcripts up to this size are given to the LLM in full

    # Storage
    db_host: str
    db_user: str
    db_password: str
    db_database: str
    sqlite_path: Path
    vectorstore_dir: Path
    audio_dir: Path
    config_dir: Path

    @property
    def use_mysql(self) -> bool:
        """MySQL is used when DB_HOST is set, otherwise a local SQLite file."""
        return bool(self.db_host)

    def missing_keys(self) -> list[str]:
        missing = []
        if not self.assemblyai_api_key:
            missing.append("ASSEMBLYAI_API_KEY")
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        return missing


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma usage telemetry
    data_dir = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
    return Settings(
        assemblyai_api_key=os.getenv("ASSEMBLYAI_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        chunk_size=_int("CHUNK_SIZE", 800),
        chunk_overlap=_int("CHUNK_OVERLAP", 150),
        retrieval_k=_int("RETRIEVAL_K", 4),
        full_context_chars=_int("FULL_CONTEXT_CHARS", 12_000),
        db_host=os.getenv("DB_HOST", ""),
        db_user=os.getenv("DB_USER", ""),
        db_password=os.getenv("DB_PASSWORD", ""),
        db_database=os.getenv("DB_DATABASE", ""),
        sqlite_path=Path(os.getenv("SQLITE_PATH", data_dir / "voxaura.db")),
        vectorstore_dir=Path(os.getenv("VECTORSTORE_DIR", data_dir / "vectorstore")),
        audio_dir=Path(os.getenv("AUDIO_DIR", data_dir / "audio")),
        config_dir=PROJECT_ROOT / "config",
    )
