"""Application settings, read once from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _int(name: str, value: str, default: int) -> int:
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


def load_settings(env_file: Path = PROJECT_ROOT / ".env") -> Settings:
    # The .env file is re-read on every call and never copied into os.environ,
    # so editing a key takes effect on the next page refresh. (load_dotenv() would
    # keep the first value it saw until the server restarts.) Real environment
    # variables still take precedence over the file.
    file_values = {k: v for k, v in dotenv_values(env_file).items() if v is not None}

    def env(name: str, default=""):
        value = os.environ.get(name, file_values.get(name))
        return default if value is None else value.strip()

    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma usage telemetry
    data_dir = Path(env("DATA_DIR") or PROJECT_ROOT / "data")
    return Settings(
        assemblyai_api_key=env("ASSEMBLYAI_API_KEY"),
        openai_api_key=env("OPENAI_API_KEY"),
        chat_model=env("OPENAI_CHAT_MODEL") or "gpt-4o-mini",
        embedding_model=env("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small",
        chunk_size=_int("CHUNK_SIZE", env("CHUNK_SIZE"), 800),
        chunk_overlap=_int("CHUNK_OVERLAP", env("CHUNK_OVERLAP"), 150),
        retrieval_k=_int("RETRIEVAL_K", env("RETRIEVAL_K"), 4),
        full_context_chars=_int("FULL_CONTEXT_CHARS", env("FULL_CONTEXT_CHARS"), 12_000),
        db_host=env("DB_HOST"),
        db_user=env("DB_USER"),
        db_password=env("DB_PASSWORD"),
        db_database=env("DB_DATABASE"),
        sqlite_path=Path(env("SQLITE_PATH") or data_dir / "voxaura.db"),
        vectorstore_dir=Path(env("VECTORSTORE_DIR") or data_dir / "vectorstore"),
        audio_dir=Path(env("AUDIO_DIR") or data_dir / "audio"),
        config_dir=PROJECT_ROOT / "config",
    )
