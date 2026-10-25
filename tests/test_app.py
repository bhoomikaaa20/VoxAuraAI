"""UI tests: drive the real Streamlit app headlessly with Streamlit's AppTest.

The OpenAI classes are replaced by local fakes, so no API keys, network or
costs are involved.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.conftest import BagOfWordsEmbeddings, ScriptedChatModel

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TRANSCRIPT = ("Um, welcome to the launch event. The new phone has a three day battery. "
              "We are, like, really proud of the camera. Pre-orders open on Friday.")


@pytest.fixture
def app(tmp_path, monkeypatch):
    import langchain_openai

    monkeypatch.setenv("DB_HOST", "")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "test")
    monkeypatch.setattr(langchain_openai, "OpenAIEmbeddings", lambda **_: BagOfWordsEmbeddings())
    monkeypatch.setattr(langchain_openai, "ChatOpenAI", lambda **_: ScriptedChatModel())
    st.cache_resource.clear()  # get_pipeline is cached per process
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    yield at
    st.cache_resource.clear()


def add_text_session(at, text=TRANSCRIPT):
    at.segmented_control(key="input_method").set_value("📄 Text transcript").run()
    at.text_area[0].input(text).run()
    next(b for b in at.button if b.label == "Analyse transcript").click().run()
    assert not at.exception, at.exception
    return at


def nav(at, page):
    at.segmented_control(key="nav").set_value(page).run()
    assert not at.exception, at.exception


def test_text_transcript_goes_through_the_whole_app(app):
    add_text_session(app)
    assert not app.error, [e.value for e in app.error]
    assert app.session_state["nav"] == "✨ Insights"  # jumps to the results
    assert any("Phone launch" in b.label for b in app.sidebar.button)
    assert [c.label for c in app.checkbox] == ["Open pre-orders on Friday"]

    nav(app, "📝 Transcript")
    app.segmented_control(key="transcript_view").set_value("Original + fillers").run()
    assert any(":orange-background[Um]" in m.value for m in app.markdown)
    app.segmented_control(key="transcript_view").set_value("Timestamped").run()
    assert not app.exception


def test_suggested_questions_answer_and_stay_visible(app):
    add_text_session(app)
    nav(app, "💬 Ask AI")
    labels = [b.label for b in app.button if b.key and b.key.startswith("suggest-")]
    assert labels[0] == "How long does the battery last?"  # generated for this recording

    next(b for b in app.button if b.label == labels[0]).click().run()
    assert not app.exception and not app.error
    assert [m.markdown[0].value for m in app.chat_message] == [labels[0], f"Answer to: {labels[0]} [1]"]

    # suggestions are still shown, minus the one already asked, and the next one answers too
    remaining = [b.label for b in app.button if b.key and b.key.startswith("suggest-")]
    assert remaining and labels[0] not in remaining
    next(b for b in app.button if b.label == remaining[0]).click().run()
    assert len(app.chat_message) == 4


def test_switching_sessions_restores_each_chat(app):
    add_text_session(app)
    nav(app, "💬 Ask AI")
    app.chat_input[0].set_value("When do pre-orders open?").run()
    first_id = app.session_state["session"]["session_id"]

    nav(app, "🎧 Add audio")
    add_text_session(app, "A second recording about gardening. Tomatoes need sun and water daily.")
    second_id = app.session_state["session"]["session_id"]
    assert app.session_state["messages"] == []

    app.sidebar.button(key=f"session-{first_id}").click().run()
    assert not app.exception
    assert app.session_state["session"]["session_id"] == first_id != second_id
    assert app.session_state["messages"][0]["content"] == "When do pre-orders open?"


def test_every_input_method_and_empty_page_renders(app):
    for method in ["📁 Upload audio", "🎙️ Record live", "▶️ YouTube link", "📄 Text transcript"]:
        app.segmented_control(key="input_method").set_value(method).run()
        assert not app.exception, (method, app.exception)
    assert len(app.audio_input) == 0  # only rendered on the recording page
    nav(app, "💬 Ask AI")
    assert any("No session open" in m.value for m in app.markdown)


def test_invalid_transcript_shows_error(app):
    add_text_session(app, "too short")
    assert any("too short" in e.value for e in app.error)
    assert app.session_state["session"] is None
