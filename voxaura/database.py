"""Persistence for sessions (transcripts + insights) and chat history.

Uses the existing MySQL database when DB_HOST is configured, otherwise a local
SQLite file so the app also runs without a database server. Both share the
same two tables; the extra columns added in this version are created
automatically on existing MySQL tables.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

# Columns added to the original transcription_sessions table.
SESSION_EXTRA_COLUMNS = {
    "title": "VARCHAR(255)",
    "source_type": "VARCHAR(32)",
    "source_name": "VARCHAR(512)",
    "raw_text": "MEDIUMTEXT",
    "segments_json": "MEDIUMTEXT",
    "filler_stats_json": "TEXT",
    "insights_json": "TEXT",
}

SCHEMA = {
    "mysql": [
        """CREATE TABLE IF NOT EXISTS transcription_sessions (
            session_id VARCHAR(255) PRIMARY KEY,
            transcription_text MEDIUMTEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE IF NOT EXISTS chat_messages (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(255),
            role ENUM('user', 'assistant') NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES transcription_sessions(session_id) ON DELETE CASCADE)""",
    ],
    "sqlite": [
        """CREATE TABLE IF NOT EXISTS transcription_sessions (
            session_id TEXT PRIMARY KEY,
            transcription_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
        """CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT REFERENCES transcription_sessions(session_id),
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""",
    ],
}


class DatabaseError(RuntimeError):
    pass


class Database:
    def __init__(self, backend: str, mysql_config: dict | None = None, sqlite_path: Path | None = None):
        if backend not in SCHEMA:
            raise ValueError(f"Unknown database backend {backend!r}")
        self.backend = backend
        self.mysql_config = mysql_config or {}
        self.sqlite_path = sqlite_path
        self.ph = "%s" if backend == "mysql" else "?"  # SQL parameter placeholder
        self.init_schema()

    @classmethod
    def from_settings(cls, settings) -> "Database":
        if settings.use_mysql:
            return cls("mysql", mysql_config={
                "host": settings.db_host, "user": settings.db_user,
                "password": settings.db_password, "database": settings.db_database,
            })
        return cls("sqlite", sqlite_path=settings.sqlite_path)

    @property
    def description(self) -> str:
        if self.backend == "mysql":
            return f"MySQL ({self.mysql_config.get('host')})"
        return f"SQLite ({self.sqlite_path.name})"

    @contextmanager
    def _cursor(self):
        try:
            if self.backend == "mysql":
                import mysql.connector

                # A small pool reuses connections instead of opening a new one per
                # query, which made switching sessions noticeably slow.
                conn = mysql.connector.connect(**self.mysql_config, connection_timeout=10,
                                               pool_name="voxaura", pool_size=5)
            else:
                self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self.sqlite_path)
        except Exception as e:
            raise DatabaseError(f"Could not connect to {self.backend} database: {e}") from e
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise DatabaseError(f"Database error: {e}") from e
        finally:
            conn.close()

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(sql.replace("?", self.ph), params)
            columns = [c[0] for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._cursor() as cur:
            cur.execute(sql.replace("?", self.ph), params)

    def init_schema(self) -> None:
        for statement in SCHEMA[self.backend]:
            self._execute(statement)
        with self._cursor() as cur:
            cur.execute("SELECT * FROM transcription_sessions WHERE 1 = 0")
            existing = {c[0] for c in cur.description}
            cur.fetchall()
        for column, col_type in SESSION_EXTRA_COLUMNS.items():
            if column not in existing:
                self._execute(f"ALTER TABLE transcription_sessions ADD COLUMN {column} {col_type}")
        with self._cursor() as cur:
            cur.execute("SELECT * FROM chat_messages WHERE 1 = 0")
            existing = {c[0] for c in cur.description}
            cur.fetchall()
        if "sources_json" not in existing:
            self._execute("ALTER TABLE chat_messages ADD COLUMN sources_json TEXT")

    # ---------------------------- sessions ---------------------------- #

    def save_session(self, session: dict) -> None:
        self._execute(
            """INSERT INTO transcription_sessions
               (session_id, title, source_type, source_name, transcription_text, raw_text,
                segments_json, filler_stats_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session["session_id"], session["title"], session["source_type"], session["source_name"],
             session["text"], session["raw_text"], json.dumps(session["segments"]),
             json.dumps(session["filler_stats"])),
        )

    def update_insights(self, session_id: str, insights: dict, title: str | None = None) -> None:
        if title:
            self._execute("UPDATE transcription_sessions SET insights_json = ?, title = ? WHERE session_id = ?",
                          (json.dumps(insights), title, session_id))
        else:
            self._execute("UPDATE transcription_sessions SET insights_json = ? WHERE session_id = ?",
                          (json.dumps(insights), session_id))

    def list_sessions(self) -> list[dict]:
        rows = self._query(
            """SELECT ts.session_id, ts.title, ts.source_type, ts.transcription_text, ts.created_at,
                      COUNT(cm.id) AS message_count
               FROM transcription_sessions ts
               LEFT JOIN chat_messages cm ON ts.session_id = cm.session_id
               GROUP BY ts.session_id, ts.title, ts.source_type, ts.transcription_text, ts.created_at
               ORDER BY ts.created_at DESC"""
        )
        for row in rows:
            # Sessions created by the old app have no title - use the first words instead.
            row["title"] = row["title"] or " ".join(row.pop("transcription_text").split()[:8]) + "..."
            row.pop("transcription_text", None)
            row["created_at"] = str(row["created_at"])[:16]
        return rows

    def get_session(self, session_id: str) -> dict | None:
        rows = self._query("SELECT * FROM transcription_sessions WHERE session_id = ?", (session_id,))
        if not rows:
            return None
        row = rows[0]
        return {
            "session_id": row["session_id"],
            "title": row["title"] or "Untitled session",
            "source_type": row["source_type"] or "legacy",
            "source_name": row["source_name"] or "",
            "text": row["transcription_text"],
            "raw_text": row["raw_text"] or row["transcription_text"],
            "segments": json.loads(row["segments_json"]) if row["segments_json"] else None,
            "filler_stats": json.loads(row["filler_stats_json"]) if row["filler_stats_json"] else None,
            "insights": json.loads(row["insights_json"]) if row["insights_json"] else None,
            "created_at": str(row["created_at"])[:16],
        }

    def delete_session(self, session_id: str) -> None:
        self._execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        self._execute("DELETE FROM transcription_sessions WHERE session_id = ?", (session_id,))

    # ------------------------------ chat ------------------------------ #

    def add_message(self, session_id: str, role: str, content: str, sources: list | None = None) -> None:
        self._execute(
            "INSERT INTO chat_messages (session_id, role, content, sources_json) VALUES (?, ?, ?, ?)",
            (session_id, role, content, json.dumps(sources) if sources else None),
        )

    def get_messages(self, session_id: str) -> list[dict]:
        rows = self._query(
            "SELECT role, content, sources_json FROM chat_messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [
            {"role": r["role"], "content": r["content"],
             "sources": json.loads(r["sources_json"]) if r["sources_json"] else []}
            for r in rows
        ]
