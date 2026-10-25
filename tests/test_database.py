import sqlite3

from voxaura.database import Database


def sample_session(session_id="abc123"):
    return {
        "session_id": session_id, "title": "Team meeting", "source_type": "upload",
        "source_name": "meeting.mp3", "text": "Clean text.", "raw_text": "Um, clean text.",
        "segments": [{"text": "Clean text.", "start_ms": 0, "end_ms": 900}],
        "filler_stats": {"counts": {"um": 1}, "total_words": 3},
    }


def test_session_round_trip(tmp_path):
    db = Database("sqlite", sqlite_path=tmp_path / "t.db")
    db.save_session(sample_session())

    loaded = db.get_session("abc123")
    assert loaded["text"] == "Clean text."
    assert loaded["segments"][0]["end_ms"] == 900
    assert loaded["filler_stats"]["counts"] == {"um": 1}
    assert loaded["insights"] is None

    db.update_insights("abc123", {"summary": "S"}, title="Better title")
    assert db.get_session("abc123")["insights"] == {"summary": "S"}
    assert db.list_sessions()[0]["title"] == "Better title"


def test_chat_messages_with_sources(tmp_path):
    db = Database("sqlite", sqlite_path=tmp_path / "t.db")
    db.save_session(sample_session())
    db.add_message("abc123", "user", "Question?")
    db.add_message("abc123", "assistant", "Answer [1].", [{"chunk_index": 0, "similarity": 0.9}])

    messages = db.get_messages("abc123")
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["sources"][0]["chunk_index"] == 0
    assert db.list_sessions()[0]["message_count"] == 2

    db.delete_session("abc123")
    assert db.get_session("abc123") is None
    assert db.get_messages("abc123") == []


def test_upgrades_tables_from_previous_version(tmp_path):
    """Sessions saved by the original app (fewer columns) still load."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE transcription_sessions (session_id TEXT PRIMARY KEY, "
                 "transcription_text TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE chat_messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
                 "role TEXT, content TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("INSERT INTO transcription_sessions (session_id, transcription_text) VALUES "
                 "('old1', 'one two three four five six seven eight nine ten')")
    conn.commit()
    conn.close()

    db = Database("sqlite", sqlite_path=path)
    assert db.list_sessions()[0]["title"] == "one two three four five six seven eight..."
    old = db.get_session("old1")
    assert old["segments"] is None and old["source_type"] == "legacy"
    db.add_message("old1", "user", "hi")  # new sources_json column exists
