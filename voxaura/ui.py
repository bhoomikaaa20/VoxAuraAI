"""Streamlit UI components. Each page of the app is rendered by one function here.

Only the active page is rendered on each rerun (rather than every tab), which
keeps switching sessions and pages fast.
"""

from __future__ import annotations

import html
from collections import Counter
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

from .audio_input import (SUPPORTED_AUDIO_TYPES, AudioInputError, check_recording, download_youtube_audio,
                          save_audio_bytes)
from .database import DatabaseError
from .insights import Insights, InsightsParseError
from .pipeline import VoxAuraPipeline
from .rag import cited_numbers
from .text_cleaning import FillerStats
from .transcript_files import SUPPORTED_TEXT_TYPES, TranscriptFileError
from .transcription import TranscriptionError, format_timestamp

USER_AVATAR = "👤"
BOT_AVATAR = "🔉"
PAGES = ["🎧 Add audio", "📝 Transcript", "✨ Insights", "💬 Ask AI"]
INPUT_PAGE, TRANSCRIPT_PAGE, INSIGHTS_PAGE, CHAT_PAGE = PAGES
SOURCE_ICONS = {"upload": "📁", "recording": "🎙️", "youtube": "▶️", "text": "📄", "legacy": "🗂️"}
SOURCE_LABELS = {"upload": "Uploaded audio", "recording": "Live recording", "youtube": "YouTube",
                 "text": "Text transcript", "legacy": "Earlier version"}
# Used when a session has no AI-generated suggestions yet. These are broad,
# analytical questions that the RAG pipeline answers from the whole recording.
DEFAULT_SUGGESTIONS = [
    "What is the main topic of this recording?",
    "How would you describe the speaker's tone and sentiment?",
    "What is the speaker trying to achieve, and how?",
    "What are the key arguments or points made?",
]


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #


def init_state() -> None:
    state = st.session_state
    state.setdefault("session", None)  # the active session dict
    state.setdefault("messages", [])  # chat history for the active session
    state.setdefault("notices", [])  # (kind, text) messages shown once after a rerun
    state.setdefault("sessions", None)  # cached sidebar list; None = reload from the database
    state.setdefault("nav", INPUT_PAGE)
    state.setdefault("last_page", INPUT_PAGE)
    # A page change requested by code is applied here, before the navigation
    # widget exists (Streamlit does not allow changing a widget's value after).
    if "_goto" in state:
        state["nav"] = state["last_page"] = state.pop("_goto")


def goto(page: str) -> None:
    st.session_state["_goto"] = page


def notify(kind: str, text: str) -> None:
    st.session_state["notices"].append((kind, text))


def open_session(pipeline: VoxAuraPipeline, session: dict) -> None:
    st.session_state["session"] = session
    st.session_state["messages"] = pipeline.db.get_messages(session["session_id"])
    st.session_state["sessions"] = None
    for warning in session.pop("warnings", []):
        notify("warning", warning)


def clear_session() -> None:
    st.session_state["session"] = None
    st.session_state["messages"] = []


def esc(text: str) -> str:
    """Escape text for HTML blocks (uploaded transcripts may contain '<')."""
    return html.escape(text or "")


def escape_md(text: str) -> str:
    # "$" would otherwise be rendered as LaTeX by Streamlit.
    return (text or "").replace("$", "\\$")


def friendly_error(e: Exception) -> str:
    """Turn common exceptions into messages that say what to do next."""
    if isinstance(e, (AudioInputError, TranscriptionError, TranscriptFileError, DatabaseError, ValueError)):
        return str(e)
    name = type(e).__name__
    if name == "AuthenticationError":
        return "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env file."
    if name == "RateLimitError":
        return "OpenAI rate limit or quota reached. Wait a moment, or check your OpenAI billing."
    if name in ("APIConnectionError", "APITimeoutError") or isinstance(e, requests.ConnectionError):
        return "Could not reach the AI service. Check your internet connection and try again."
    return f"Unexpected error ({name}): {e}"


def session_metrics(session: dict) -> dict:
    segments = session.get("segments") or []
    stats = FillerStats.from_dict(session.get("filler_stats"))
    end_ms = segments[-1].get("end_ms") if segments else None
    minutes = end_ms / 60_000 if end_ms else None
    return {
        "words": len(session["text"].split()),
        "duration": format_timestamp(end_ms) if end_ms else "—",
        "wpm": round(stats.total_words / minutes) if minutes and minutes >= 0.25 else None,
        "fillers": stats,
        "sentiment": Counter(s.get("sentiment") for s in segments if s.get("sentiment")),
    }


# --------------------------------------------------------------------------- #
# Layout pieces
# --------------------------------------------------------------------------- #


def load_css() -> None:
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    session = st.session_state["session"]
    active = ""
    if session:
        m = session_metrics(session)
        icon = SOURCE_ICONS.get(session["source_type"], "🗂️")
        active = (f'<div class="vox-active"><span class="vox-chip">{icon} {esc(session["title"])}</span>'
                  f'<span class="vox-chip">{m["words"]} words</span>'
                  f'<span class="vox-chip">⏱ {m["duration"]}</span></div>')
    st.markdown(
        '<div class="vox-hero"><h1>🔉 VoxAuraAI</h1>'
        "<p>Transcribe speech, surface insights, and ask questions answered from the recording itself.</p>"
        f"{active}</div>",
        unsafe_allow_html=True,
    )
    for kind, text in st.session_state["notices"]:
        getattr(st, kind)(text)
    st.session_state["notices"] = []


def render_nav() -> str:
    def keep_selection():
        # A segmented control can be clicked "off"; keep the previous page instead.
        if st.session_state["nav"] is None:
            st.session_state["nav"] = st.session_state["last_page"]
        st.session_state["last_page"] = st.session_state["nav"]

    st.segmented_control("Navigation", PAGES, key="nav", on_change=keep_selection,
                         label_visibility="collapsed")
    return st.session_state["nav"] or INPUT_PAGE


def empty_state(icon: str, title: str, text: str) -> None:
    st.markdown(f'<div class="vox-empty"><div class="icon">{icon}</div><h3>{title}</h3><p>{text}</p></div>',
                unsafe_allow_html=True)


def stat_cards(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(f'<div class="vox-stat"><div class="label">{label}</div><div class="value">{value}</div>'
                    f'<div class="hint">{hint}</div></div>' for label, value, hint in items)
    st.markdown(f'<div class="vox-stats">{cards}</div>', unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="vox-card"><h4>{title}</h4>{body_html}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar: sessions and settings
# --------------------------------------------------------------------------- #


def _select_session(pipeline: VoxAuraPipeline, session_id: str) -> None:
    """Button callback: runs before the page is redrawn, so the new session appears immediately."""
    try:
        session = pipeline.db.get_session(session_id)
    except DatabaseError as e:
        notify("error", str(e))
        return
    if session is None:
        notify("error", "That session no longer exists.")
        st.session_state["sessions"] = None
        return
    open_session(pipeline, session)
    if st.session_state["nav"] == INPUT_PAGE:
        goto(INSIGHTS_PAGE)


def _new_session() -> None:
    clear_session()
    goto(INPUT_PAGE)


def render_sidebar(pipeline: VoxAuraPipeline) -> None:
    with st.sidebar:
        st.markdown('<div class="vox-brand">🔉 VoxAuraAI</div>', unsafe_allow_html=True)
        st.button("➕ New session", use_container_width=True, type="primary", on_click=_new_session)

        if st.session_state["sessions"] is None:
            try:
                st.session_state["sessions"] = pipeline.db.list_sessions()
            except DatabaseError as e:
                st.error(str(e))
                st.session_state["sessions"] = []
        sessions = st.session_state["sessions"]

        st.markdown("##### History")
        query = st.text_input("Search sessions", placeholder="🔍 Search sessions",
                              label_visibility="collapsed") if len(sessions) > 6 else ""
        active_id = (st.session_state["session"] or {}).get("session_id")
        shown = [s for s in sessions if query.lower() in s["title"].lower()]
        if not sessions:
            st.caption("No sessions yet - add some audio to get started.")
        with st.container(height=min(60 + 44 * len(shown), 420), border=False):
            for s in shown:
                icon = SOURCE_ICONS.get(s["source_type"] or "legacy", "🗂️")
                st.button(f"{icon} {s['title']}", key=f"session-{s['session_id']}", use_container_width=True,
                          type="primary" if s["session_id"] == active_id else "secondary",
                          help=f"{s['created_at']} · {s['message_count']} messages",
                          on_click=_select_session, args=(pipeline, s["session_id"]))

        if st.session_state["session"]:
            with st.popover("🗑️ Delete current session", use_container_width=True):
                st.write("This permanently removes the transcript, insights, chat and vector index.")
                if st.button("Yes, delete it", type="primary"):
                    with st.spinner("Deleting..."):
                        pipeline.delete_session(st.session_state["session"])
                    clear_session()
                    st.session_state["sessions"] = None
                    goto(INPUT_PAGE)
                    st.rerun()

        with st.expander("⚙️ Configuration"):
            settings = pipeline.settings
            st.markdown(
                f"- **Database:** {pipeline.db.description}\n"
                f"- **LLM:** `{settings.chat_model}`\n"
                f"- **Embeddings:** `{settings.embedding_model}`\n"
                f"- **Chunks:** {settings.chunk_size} chars, {settings.chunk_overlap} overlap\n"
                f"- **Retrieved per question (k):** {settings.retrieval_k}\n"
                f"- **Full-transcript context up to:** {settings.full_context_chars:,} chars"
            )


# --------------------------------------------------------------------------- #
# Page 1: add audio
# --------------------------------------------------------------------------- #

PIPELINE_STEPS = 7  # roughly the number of progress messages the pipeline sends


def _run_pipeline(pipeline: VoxAuraPipeline, process) -> None:
    """Run a pipeline function with a progress bar, a step log and clear error messages."""
    bar = st.progress(0.0, text="Starting...")
    with st.status("Processing...", expanded=True) as status:
        step = [0]

        def progress(message: str) -> None:
            step[0] += 1
            status.write(message)
            bar.progress(min(step[0] / PIPELINE_STEPS, 0.95), text=message)

        try:
            session = process(progress)
        except Exception as e:
            status.update(label="Processing failed", state="error", expanded=True)
            bar.empty()
            st.error(friendly_error(e))
            return
        bar.progress(1.0, text="Done")
        status.update(label="Done!", state="complete", expanded=False)
    open_session(pipeline, session)
    notify("success", f"**{session['title']}** is ready - here are the insights. "
                      "Open **Ask AI** to ask questions about it.")
    goto(INSIGHTS_PAGE)
    st.rerun()


def render_input_page(pipeline: VoxAuraPipeline) -> None:
    audio_dir = pipeline.settings.audio_dir
    st.markdown("#### Add a recording")
    method = st.segmented_control(
        "Input method", ["📁 Upload audio", "🎙️ Record live", "▶️ YouTube link", "📄 Text transcript"],
        default="📁 Upload audio", key="input_method", label_visibility="collapsed",
    ) or "📁 Upload audio"

    with st.container(border=True):
        if method == "📁 Upload audio":
            st.caption(f"Supported: {', '.join(SUPPORTED_AUDIO_TYPES)} - up to 200 MB.")
            uploaded = st.file_uploader("Audio file", type=SUPPORTED_AUDIO_TYPES, label_visibility="collapsed")
            if uploaded:
                st.audio(uploaded)
                if st.button("Transcribe & analyse", type="primary"):
                    _run_pipeline(pipeline, lambda p: pipeline.process_audio(
                        save_audio_bytes(uploaded.getvalue(), uploaded.name, audio_dir),
                        "upload", uploaded.name, p))

        elif method == "🎙️ Record live":
            st.caption("Click the microphone to start, and click the stop button when you finish. "
                       "Your browser will ask for microphone permission. It only works on "
                       "localhost or HTTPS.")
            recording = st.audio_input("Record from your microphone", label_visibility="collapsed")
            if recording:
                if st.button("Transcribe recording", type="primary"):
                    def process(progress):
                        progress("Checking the recording...")
                        path = save_audio_bytes(recording.getvalue(), "recording.wav", audio_dir)
                        seconds = check_recording(path)
                        progress(f"Recording OK ({seconds:.1f} s).")
                        return pipeline.process_audio(path, "recording",
                                                      f"Recording {datetime.now():%d %b %Y, %H:%M}", progress)
                    _run_pipeline(pipeline, process)
            else:
                st.info("No recording yet. Press the microphone above to start.", icon="🎙️")

        elif method == "▶️ YouTube link":
            url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...")
            if st.button("Download & transcribe", type="primary", disabled=not url.strip()):
                def process(progress):
                    progress("Downloading the audio track...")
                    path, title = download_youtube_audio(url.strip(), audio_dir)
                    return pipeline.process_audio(path, "youtube", title, progress)
                _run_pipeline(pipeline, process)

        else:
            st.caption("Already have a transcript? Upload a .txt, .md, .srt or .vtt file (subtitle timestamps "
                       "are kept), or paste the text. Speech-to-text is skipped.")
            uploaded = st.file_uploader("Transcript file", type=SUPPORTED_TEXT_TYPES)
            pasted = st.text_area("...or paste the transcript", height=160,
                                  placeholder="Paste a transcript here (at least a few sentences).")
            if uploaded and pasted.strip():
                st.caption("Both a file and pasted text were given - the file will be used.")
            if st.button("Analyse transcript", type="primary", disabled=not (uploaded or pasted.strip())):
                if uploaded:
                    _run_pipeline(pipeline, lambda p: pipeline.process_text(uploaded.name, data=uploaded.getvalue(),
                                                                            progress=p))
                else:
                    _run_pipeline(pipeline, lambda p: pipeline.process_text("Pasted transcript", text=pasted,
                                                                            progress=p))

    st.markdown("")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1 · Transcribe**  \nAssemblyAI speech-to-text with timestamps and sentence sentiment.")
    c2.markdown("**2 · Understand**  \nFillers removed, abbreviations explained, summary, tone and action items.")
    c3.markdown("**3 · Ask**  \nRAG chat grounded in the transcript, with cited, timestamped sources.")


# --------------------------------------------------------------------------- #
# Page 2: transcript
# --------------------------------------------------------------------------- #


def render_transcript_page(pipeline: VoxAuraPipeline, session: dict) -> None:
    stats = FillerStats.from_dict(session.get("filler_stats"))
    st.markdown(f"#### {esc(session['title'])}")
    st.caption(f"{SOURCE_LABELS.get(session['source_type'], session['source_type'])} · "
               f"{len(session['text'].split())} words · {session.get('created_at', 'just now')}")

    left, right = st.columns([3, 2])
    with left:
        view = st.segmented_control("View", ["Clean", "Timestamped", "Original + fillers"], default="Clean",
                                    key="transcript_view", label_visibility="collapsed") or "Clean"
    with right:
        query = st.text_input("Find", placeholder="🔍 Find in transcript", label_visibility="collapsed")

    segments = session.get("segments") or [{"text": session["text"]}]
    with st.container(height=440, border=True):
        if view == "Original + fillers":
            st.caption(f"{stats.total_fillers} filler words detected (highlighted). They are removed from the "
                       "clean version.")
            st.markdown(pipeline.cleaner.highlight_fillers(escape_md(session["raw_text"])))
        elif view == "Clean" and not query:
            st.markdown(escape_md(session["text"]))
        else:
            rows = [s for s in segments if query.lower() in s["text"].lower()]
            if not rows:
                st.caption("No matching sentences.")
            has_time = segments[0].get("start_ms") is not None
            if view == "Timestamped" and not has_time:
                st.caption("This transcript has no timestamps (pasted text or a .txt file).")
            html_rows = "".join(
                '<div class="vox-row">'
                + (f'<span class="vox-time">{format_timestamp(s.get("start_ms"))}</span>' if has_time else "")
                + (f'<span class="vox-dot {s["sentiment"]}" title="{s["sentiment"].lower()}"></span>'
                   if s.get("sentiment") else "")
                + f"<span>{esc(s['text'])}</span></div>"
                for s in rows
            )
            st.markdown(html_rows, unsafe_allow_html=True)

    st.download_button("⬇️ Download clean transcript", session["text"],
                       file_name=f"transcript_{session['session_id']}.txt")

    with st.expander("🔍 How this transcript is indexed for RAG"):
        chunks = pipeline.chunks_for(session)
        st.caption(f"{len(chunks)} chunks · sentences grouped into ~{pipeline.settings.chunk_size}-character "
                   f"chunks with ~{pipeline.settings.chunk_overlap} characters of overlap, embedded with "
                   f"`{pipeline.settings.embedding_model}` and stored in this session's Chroma collection.")
        st.markdown("".join(
            f'<div class="vox-source"><div class="meta">Chunk {c.index} · {c.time_range} · {len(c.text)} chars'
            f"</div>{esc(c.text)}</div>" for c in chunks), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page 3: insights
# --------------------------------------------------------------------------- #


def render_insights_page(pipeline: VoxAuraPipeline, session: dict) -> None:
    m = session_metrics(session)
    stats: FillerStats = m["fillers"]
    stat_cards([
        ("Duration", m["duration"], SOURCE_LABELS.get(session["source_type"], "")),
        ("Words", str(m["words"]), "clean transcript"),
        ("Speaking rate", f"{m['wpm']} wpm" if m["wpm"] else "—", "typical speech is 120-160"),
        ("Fillers", f"{stats.per_100_words}", f"per 100 words · {stats.total_fillers} total"),
    ])

    insights = Insights.from_dict(session.get("insights"))
    head, button = st.columns([4, 1])
    head.markdown("#### ✨ AI insights")
    label = "Generate" if insights is None else "🔄 Regenerate"
    if button.button(label, use_container_width=True, type="primary" if insights is None else "secondary"):
        with st.spinner("Analysing the transcript - this takes a few seconds..."):
            try:
                pipeline.create_insights(session)
                st.session_state["sessions"] = None  # the title may have changed
            except (InsightsParseError, Exception) as e:
                st.error(f"Could not generate insights: {friendly_error(e)}")
                return
        st.rerun()

    if insights is None:
        empty_state("✨", "No insights yet", "Generate a summary, tone analysis, key points and action items.")
    else:
        card("Summary", f"<p>{esc(insights.summary)}</p>")
        if insights.tone:
            card("Tone & speaking style", f"<p>{esc(insights.tone)}</p>")
        left, right = st.columns(2)
        with left:
            items = "".join(f"<li>{esc(p)}</li>" for p in insights.key_points) or "<li>None</li>"
            card("Key points", f"<ul>{items}</ul>")
        with right:
            with st.container(border=True):
                st.markdown("**ACTION ITEMS**")
                if insights.action_items:
                    for i, item in enumerate(insights.action_items):
                        st.checkbox(item, key=f"action-{session['session_id']}-{i}")
                else:
                    st.caption("No action items mentioned.")
        st.caption(f"Generated by `{pipeline.settings.chat_model}` ({insights.method}). AI-generated - "
                   "check important details against the transcript.")
        st.download_button("⬇️ Download insights (Markdown)", insights.to_markdown(),
                           file_name=f"insights_{session['session_id']}.md")

    st.markdown("#### 📊 Speech analysis")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Filler words**")
        if stats.counts:
            st.bar_chart(stats.counts, horizontal=True, height=220, color="#6D28D9")
        else:
            st.caption("No filler words detected.")
    with right, st.container(border=True):
        st.markdown("**Sentence sentiment**")
        if m["sentiment"]:
            order = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
            st.bar_chart({k.title(): m["sentiment"].get(k, 0) for k in order}, height=220, color="#DB2777")
            st.caption("From AssemblyAI's sentiment model, per sentence.")
        else:
            st.caption("Sentiment is available for audio transcribed with this version.")


# --------------------------------------------------------------------------- #
# Page 4: chat
# --------------------------------------------------------------------------- #


def _render_sources(sources: list[dict], answer: str, meta: dict | None = None) -> None:
    if not sources:
        return
    cited = cited_numbers(answer)
    label = f"📎 Sources · {len(cited & set(range(1, len(sources) + 1)))} cited of {len(sources)} used"
    with st.expander(label):
        if meta:
            st.caption(f"Question type: **{meta.get('question_type')}** · context: {meta.get('strategy')}"
                       + (f" · searched for: _{escape_md(meta['search_query'])}_" if meta.get("rewritten") else ""))
        st.markdown("".join(
            f'<div class="vox-source{"" if i in cited else " uncited"}"><div class="meta">[{i}] {s["time_range"]} · '
            f'chunk {s["chunk_index"]} · similarity {s["similarity"]:.2f} · {"cited" if i in cited else "not cited"}'
            f'</div>{esc(s["text"])}</div>'
            for i, s in enumerate(sources, start=1)), unsafe_allow_html=True)


def suggestions_for(session: dict, messages: list[dict]) -> list[str]:
    """Session-specific suggestions first, topped up with general ones; already-asked ones are skipped."""
    asked = {m["content"].strip().lower() for m in messages if m["role"] == "user"}
    generated = (session.get("insights") or {}).get("suggested_questions") or []
    pool = [q for q in generated + DEFAULT_SUGGESTIONS if q.strip().lower() not in asked]
    return list(dict.fromkeys(pool))[:4]


def _ask_suggestion(question: str) -> None:
    st.session_state["pending_question"] = question


def _answer(pipeline: VoxAuraPipeline, session: dict, question: str) -> None:
    history = list(st.session_state["messages"])  # turns *before* this question
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(escape_md(question))
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            with st.spinner("Preparing the vector index..." if session["session_id"] not in pipeline._chats
                            else "Reading the transcript and thinking..."):
                result = pipeline.chat(session).ask(question, history)
        except Exception as e:
            st.error(f"Could not answer: {friendly_error(e)}")
            return
        sources = [{"chunk_index": r.chunk.index, "time_range": r.chunk.time_range,
                    "similarity": r.similarity, "text": r.chunk.text} for r in result.sources]
        meta = {"question_type": result.question_type, "strategy": result.strategy,
                "search_query": result.search_query, "rewritten": result.search_query != question}
        st.markdown(escape_md(result.answer))
        _render_sources(sources, result.answer, meta)

    st.session_state["messages"] += [{"role": "user", "content": question, "sources": []},
                                     {"role": "assistant", "content": result.answer, "sources": sources}]
    st.session_state["sessions"] = None  # message counts changed
    try:
        pipeline.db.add_message(session["session_id"], "user", question)
        pipeline.db.add_message(session["session_id"], "assistant", result.answer, sources)
    except DatabaseError as e:
        st.warning(f"The answer could not be saved to history: {e}")


def render_chat_page(pipeline: VoxAuraPipeline, session: dict) -> None:
    messages = st.session_state["messages"]
    if not messages:
        empty_state("💬", "Ask anything about this recording",
                    "Facts, tone, intent, arguments... Answers cite the transcript and keep evidence "
                    "separate from interpretation.")
        st.markdown("")

    # The question is read first so the suggestions below can already leave it out.
    # (st.chat_input is pinned to the bottom of the page wherever it is called.)
    question = st.chat_input("Ask a question about the recording...") or st.session_state.pop("pending_question", None)

    conversation = st.container()
    st.caption("Suggested questions")
    asked = messages + ([{"role": "user", "content": question}] if question else [])
    cols = st.columns(2)
    for i, q in enumerate(suggestions_for(session, asked)):
        cols[i % 2].button(q, key=f"suggest-{i}-{q}", use_container_width=True,
                           on_click=_ask_suggestion, args=(q,))

    with conversation:
        for m in messages:
            with st.chat_message(m["role"], avatar=USER_AVATAR if m["role"] == "user" else BOT_AVATAR):
                st.markdown(escape_md(m["content"]))
                _render_sources(m.get("sources") or [], m["content"])
        if question:
            _answer(pipeline, session, question)
