"""Streamlit UI components. Each page of the app is rendered by one function here.

Only the active page is rendered on each rerun (rather than every tab), which
keeps switching sessions and pages fast. Visual styling lives in styles.css.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from ..analysis.insights import Insights, InsightsParseError
from ..analysis.rag import cited_numbers
from ..database import DatabaseError
from ..errors import friendly_error
from ..ingestion.audio_input import SUPPORTED_AUDIO_TYPES, check_recording, download_youtube_audio, save_audio_bytes
from ..ingestion.transcript_files import SUPPORTED_TEXT_TYPES
from ..ingestion.transcription import format_timestamp
from ..pipeline import VoxAuraPipeline, session_metrics

USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/graphic_eq:"

INPUT_PAGE, INSIGHTS_PAGE, TRANSCRIPT_PAGE, CHAT_PAGE = "new", "overview", "transcript", "chat"
NAV_ITEMS = [  # (page, label, icon) shown under "Workspace" in the sidebar
    (INSIGHTS_PAGE, "Overview", ":material/insights:"),
    (TRANSCRIPT_PAGE, "Transcript", ":material/subject:"),
    (CHAT_PAGE, "Ask AI", ":material/forum:"),
]
SOURCE_ICONS = {"upload": ":material/upload_file:", "recording": ":material/mic:", "youtube": ":material/smart_display:",
                "text": ":material/description:", "legacy": ":material/history:"}
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
PRIMARY, ACCENT = "#6D28D9", "#EC4899"
SENTIMENT_COLOURS = {"Positive": "#16A34A", "Neutral": "#B4AFC4", "Negative": "#DC2626"}

# Small inline SVG icons (Lucide-style strokes) for the HTML parts of the UI.
_ICON_PATHS = {
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "text": '<path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h6"/>',
    "gauge": '<path d="M12 14l4-4"/><path d="M3.3 19a10 10 0 1 1 17.4 0"/>',
    "filler": '<path d="M12 3a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V6a3 3 0 0 0-3-3z"/><path d="M19 11a7 7 0 0 1-14 0M12 18v3"/><path d="M3 3l18 18"/>',
    "spark": '<path d="M12 3l1.8 5.4L19 10l-5.2 1.6L12 17l-1.8-5.4L5 10l5.2-1.6z"/><path d="M19 17l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1"/><circle cx="4.5" cy="12" r="1"/><circle cx="4.5" cy="18" r="1"/>',
    "smile": '<circle cx="12" cy="12" r="9"/><path d="M8.5 14.5s1.3 2 3.5 2 3.5-2 3.5-2M9 9.5h.01M15 9.5h.01"/>',
    "chat": '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20.5l1.4-4.6A8 8 0 1 1 21 12z"/>',
    "wave": '<path d="M3 12h2M7 8v8M11 5v14M15 8v8M19 10v4M21 12h0"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/>',
}


def icon(name: str, size: int = 18) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{_ICON_PATHS[name]}</svg>')


LOGO = ('<svg width="36" height="36" viewBox="0 0 36 36"><defs><linearGradient id="vxg" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#8B5CF6"/><stop offset="1" stop-color="#EC4899"/></linearGradient></defs>'
        '<rect width="36" height="36" rx="10" fill="url(#vxg)"/><g stroke="#fff" stroke-width="2.6" stroke-linecap="round">'
        '<path d="M9 16v4M13.5 12v12M18 9v18M22.5 13v10M27 16.5v3"/></g></svg>')


# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #


def init_state() -> None:
    state = st.session_state
    state.setdefault("session", None)  # the active session dict
    state.setdefault("messages", [])  # chat history for the active session
    state.setdefault("notices", [])  # (kind, text) messages shown once after a rerun
    state.setdefault("sessions", None)  # cached sidebar list; None = reload from the database
    state.setdefault("page", INPUT_PAGE)


def sync_url(pipeline: VoxAuraPipeline) -> None:
    """Keep ?page=...&session=... in the address bar, so a view can be bookmarked or refreshed."""
    state, params = st.session_state, st.query_params
    if not state.get("_url_loaded"):  # first run in this browser tab: restore from the URL
        state["_url_loaded"] = True
        session_id, page = params.get("session"), params.get("page")
        if session_id and state["session"] is None:
            try:
                session = pipeline.db.get_session(session_id)
            except DatabaseError:
                session = None
            if session:
                open_session(pipeline, session)
        if page in {INPUT_PAGE, INSIGHTS_PAGE, TRANSCRIPT_PAGE, CHAT_PAGE}:
            goto(page)
    wanted = {"page": state["page"]}
    if state["session"]:
        wanted["session"] = state["session"]["session_id"]
    if params.to_dict() != wanted:
        params.from_dict(wanted)


def goto(page: str) -> None:
    st.session_state["page"] = page


def notify(kind: str, text: str) -> None:
    st.session_state["notices"].append((kind, text))


def open_session(pipeline: VoxAuraPipeline, session: dict) -> None:
    st.session_state["session"] = pipeline.refresh_filler_stats(session)
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


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Shared layout pieces
# --------------------------------------------------------------------------- #


def load_css() -> None:
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    html_block(f"<style>{css}</style>")


def show_notices() -> None:
    for kind, text in st.session_state["notices"]:
        if kind == "success":
            st.toast(text, icon=":material/check_circle:")
        else:
            getattr(st, kind)(text)
    st.session_state["notices"] = []


def page_header(eyebrow: str, title: str, subtitle: str = "", actions=None) -> None:
    """Title block; `actions` is an optional function that draws buttons on the right."""
    heading = (f'<div class="vx-eyebrow">{esc(eyebrow)}</div><div class="vx-title">{esc(title)}</div>'
               + (f'<p class="vx-sub">{subtitle}</p>' if subtitle else ""))
    if not actions:
        html_block(heading)
    else:
        left, right = st.columns([3, 2], vertical_alignment="bottom")
        with left:
            html_block(heading)
        with right:
            actions()
    st.markdown("")


def session_subtitle(session: dict) -> str:
    m = session_metrics(session)
    parts = [SOURCE_LABELS.get(session["source_type"], "Session"), f'{m["words"]:,} words']
    if m["duration"] != "—":
        parts.append(m["duration"])
    parts.append(session.get("created_at") or "just now")
    return " &nbsp;·&nbsp; ".join(esc(p) for p in parts)


def empty_state(icon_name: str, title: str, text: str) -> None:
    html_block(f'<div class="vx-empty"><div class="ic">{icon(icon_name, 26)}</div><h3>{title}</h3><p>{text}</p></div>')


def kpis(items: list[tuple[str, str, str, str]]) -> None:
    """Row of KPI tiles: (icon, label, value, hint)."""
    html_block('<div class="vx-kpis">' + "".join(
        f'<div class="vx-kpi"><div class="ic">{icon(ic)}</div><div><div class="l">{esc(label)}</div>'
        f'<div class="v">{esc(value)}</div><div class="h">{esc(hint)}</div></div></div>'
        for ic, label, value, hint in items) + "</div>")


def card(icon_name: str, title: str, body_html: str) -> None:
    html_block(f'<div class="vx-card"><div class="vx-card-h"><div class="ic">{icon(icon_name, 16)}</div>'
               f'<div class="t">{esc(title)}</div></div>{body_html}</div>')


def card_title(icon_name: str, title: str) -> None:
    html_block(f'<div class="vx-card-h"><div class="ic">{icon(icon_name, 16)}</div><div class="t">{esc(title)}</div></div>')


# --------------------------------------------------------------------------- #
# Sidebar: brand, navigation, history, status
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
    if st.session_state["page"] == INPUT_PAGE:
        goto(INSIGHTS_PAGE)


def _new_session() -> None:
    clear_session()
    goto(INPUT_PAGE)


def render_sidebar(pipeline: VoxAuraPipeline) -> None:
    page = st.session_state["page"]
    with st.sidebar:
        html_block(f'<div class="vx-brand">{LOGO}<div><div class="name">VoxAura</div>'
                   '<div class="tag">Audio intelligence studio</div></div></div>')
        with st.container(key="newbtn"):
            st.button("New analysis", icon=":material/add:", width="stretch", on_click=_new_session,
                      key="nav-new")

        html_block('<div class="vx-side-label">Workspace</div>')
        with st.container(key="sidenav"):
            for target, label, ic in NAV_ITEMS:
                st.button(label, icon=ic, key=f"nav-{target}", width="stretch",
                          type="primary" if page == target else "secondary", on_click=goto, args=(target,))

        if st.session_state["sessions"] is None:
            try:
                st.session_state["sessions"] = pipeline.db.list_sessions()
            except DatabaseError as e:
                st.error(str(e))
                st.session_state["sessions"] = []
        sessions = st.session_state["sessions"]

        html_block(f'<div class="vx-side-label">Recent sessions · {len(sessions)}</div>')
        query = st.text_input("Search sessions", placeholder="Search sessions", icon=":material/search:",
                              label_visibility="collapsed") if len(sessions) > 5 else ""
        shown = [s for s in sessions if query.lower() in s["title"].lower()]
        active_id = (st.session_state["session"] or {}).get("session_id")
        if not sessions:
            st.caption("Your analysed recordings will appear here.")
        with st.container(key="history", height=min(48 + 42 * len(shown), 330), border=False):
            for s in shown:
                st.button(s["title"], icon=SOURCE_ICONS.get(s["source_type"] or "legacy"),
                          key=f"session-{s['session_id']}", width="stretch",
                          type="primary" if s["session_id"] == active_id else "secondary",
                          help=f"{s['created_at']} · {s['message_count']} messages",
                          on_click=_select_session, args=(pipeline, s["session_id"]))

        if st.session_state["session"]:
            with st.popover("Delete session", icon=":material/delete:", width="stretch"):
                st.write("This permanently removes the transcript, insights, chat history and vector index.")
                if st.button("Delete permanently", type="primary", icon=":material/delete_forever:"):
                    with st.spinner("Deleting..."):
                        pipeline.delete_session(st.session_state["session"])
                    clear_session()
                    st.session_state["sessions"] = None
                    goto(INPUT_PAGE)
                    st.rerun()

        settings = pipeline.settings
        html_block(
            '<div class="vx-status">'
            f'<div class="row"><span class="k"><span class="vx-dot-ok"></span>Database</span>'
            f'<span class="v">{esc(pipeline.db.backend.upper())}</span></div>'
            f'<div class="row"><span class="k">Language model</span><span class="v">{esc(settings.chat_model)}</span></div>'
            f'<div class="row"><span class="k">Embeddings</span><span class="v">{esc(settings.embedding_model)}</span></div>'
            f'<div class="row"><span class="k">Retrieval</span><span class="v">k={settings.retrieval_k} · '
            f'{settings.chunk_size} chars</span></div></div>')


# --------------------------------------------------------------------------- #
# Page: new analysis
# --------------------------------------------------------------------------- #

PIPELINE_STEPS = 7  # roughly the number of progress messages the pipeline sends


def _run_pipeline(pipeline: VoxAuraPipeline, process) -> None:
    """Run a pipeline function with a progress bar, a step log and clear error messages."""
    bar = st.progress(0.0, text="Starting...")
    with st.status("Analysing your recording...", expanded=True) as status:
        step = [0]

        def progress(message: str) -> None:
            step[0] += 1
            status.write(message)
            bar.progress(min(step[0] / PIPELINE_STEPS, 0.95), text=message)

        try:
            session = process(progress)
        except Exception as e:
            status.update(label="Analysis failed", state="error", expanded=True)
            bar.empty()
            st.error(friendly_error(e), icon=":material/error:")
            return
        bar.progress(1.0, text="Done")
        status.update(label="Analysis complete", state="complete", expanded=False)
    open_session(pipeline, session)
    notify("success", f"{session['title']} is ready")
    goto(INSIGHTS_PAGE)
    st.rerun()


def render_input_page(pipeline: VoxAuraPipeline) -> None:
    audio_dir = pipeline.settings.audio_dir
    bars = "".join(f'<i style="height:{h}px;animation-delay:{i * .12:.2f}s"></i>'
                   for i, h in enumerate([18, 34, 52, 28, 44, 62, 36, 22, 48, 30, 16]))
    html_block(
        '<div class="vx-hero"><span class="pill">New analysis</span>'
        "<h1>Turn any conversation into insight.</h1>"
        "<p>Upload or record audio and VoxAura will transcribe it, clean it up, summarise it, and let you ask "
        "questions that are answered from the recording itself, with timestamped sources.</p>"
        f'<div class="wave">{bars}</div></div>')

    with st.container(key="panel-input"):
        method = st.segmented_control(
            "Input method", ["Upload audio", "Record live", "YouTube link", "Text transcript"],
            default="Upload audio", key="input_method", label_visibility="collapsed",
        ) or "Upload audio"

        if method == "Upload audio":
            uploaded = st.file_uploader(f"Drop an audio file ({', '.join(SUPPORTED_AUDIO_TYPES)}), up to 200 MB",
                                        type=SUPPORTED_AUDIO_TYPES)
            if uploaded:
                st.audio(uploaded)
                if st.button("Transcribe & analyse", type="primary", icon=":material/auto_awesome:"):
                    _run_pipeline(pipeline, lambda p: pipeline.process_audio(
                        save_audio_bytes(uploaded.getvalue(), uploaded.name, audio_dir),
                        "upload", uploaded.name, p))

        elif method == "Record live":
            st.caption("Click the microphone to start and the stop button to finish. Your browser will ask "
                       "for microphone permission. Recording only works on localhost or HTTPS.")
            recording = st.audio_input("Record from your microphone", label_visibility="collapsed")
            if recording:
                if st.button("Transcribe recording", type="primary", icon=":material/auto_awesome:"):
                    def process(progress):
                        progress("Checking the recording...")
                        path = save_audio_bytes(recording.getvalue(), "recording.wav", audio_dir)
                        seconds = check_recording(path)
                        progress(f"Recording OK ({seconds:.1f} s).")
                        return pipeline.process_audio(path, "recording",
                                                      f"Recording {datetime.now():%d %b %Y, %H:%M}", progress)
                    _run_pipeline(pipeline, process)

        elif method == "YouTube link":
            url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...",
                                icon=":material/link:")
            if st.button("Download & analyse", type="primary", icon=":material/auto_awesome:",
                         disabled=not url.strip()):
                def process(progress):
                    progress("Downloading the audio track...")
                    path, title = download_youtube_audio(url.strip(), audio_dir)
                    return pipeline.process_audio(path, "youtube", title, progress)
                _run_pipeline(pipeline, process)

        else:
            uploaded = st.file_uploader("Upload a transcript (.txt, .md, .srt or .vtt). Subtitle timestamps are kept.",
                                        type=SUPPORTED_TEXT_TYPES)
            pasted = st.text_area("Or paste the transcript", height=150,
                                  placeholder="Paste a transcript here (at least a few sentences).")
            if uploaded and pasted.strip():
                st.caption("Both a file and pasted text were given. The file will be used.")
            if st.button("Analyse transcript", type="primary", icon=":material/auto_awesome:",
                         disabled=not (uploaded or pasted.strip())):
                if uploaded:
                    _run_pipeline(pipeline, lambda p: pipeline.process_text(uploaded.name, data=uploaded.getvalue(),
                                                                            progress=p))
                else:
                    _run_pipeline(pipeline, lambda p: pipeline.process_text("Pasted transcript", text=pasted,
                                                                            progress=p))

    html_block(
        '<div class="vx-steps">'
        f'<div class="vx-step"><div class="n">{icon("wave")}</div><div class="t">Transcribe</div>'
        '<div class="d">AssemblyAI speech-to-text with timestamps and sentence-level sentiment.</div></div>'
        f'<div class="vx-step"><div class="n">{icon("spark")}</div><div class="t">Understand</div>'
        '<div class="d">Fillers removed, acronyms explained, summary, tone, key points and action items.</div></div>'
        f'<div class="vx-step"><div class="n">{icon("chat")}</div><div class="t">Ask</div>'
        '<div class="d">RAG chat grounded in the transcript, with cited, timestamped evidence.</div></div>'
        "</div>")


# --------------------------------------------------------------------------- #
# Page: overview (insights)
# --------------------------------------------------------------------------- #


def _filler_chart(counts: dict[str, int]):
    import altair as alt
    import pandas as pd

    data = pd.DataFrame({"filler": list(counts), "count": list(counts.values())}).head(8)
    base = alt.Chart(data).encode(
        y=alt.Y("filler:N", sort="-x", title=None,
                axis=alt.Axis(domain=False, ticks=False, labelFontSize=12, labelColor="#3D3852", labelPadding=8)),
        x=alt.X("count:Q", title=None, axis=None),
        tooltip=["filler", "count"])
    bars = base.mark_bar(cornerRadiusEnd=6, height=16, color=PRIMARY)
    labels = base.mark_text(align="left", dx=6, fontSize=12, fontWeight=600, color="#3D3852").encode(text="count:Q")
    return ((bars + labels).properties(height=max(90, 30 * len(data)))
            .configure_view(strokeWidth=0).configure(background="transparent", font="Inter"))


def _sentiment_chart(sentiment) -> object:
    import altair as alt
    import pandas as pd

    order = ["Positive", "Neutral", "Negative"]
    total = sum(sentiment.values())
    data = pd.DataFrame({"sentiment": order,
                         "count": [sentiment.get(k.upper(), 0) for k in order]})
    data["share"] = (100 * data["count"] / total).round().astype(int).astype(str) + "%"
    donut = alt.Chart(data).mark_arc(innerRadius=52, outerRadius=78, cornerRadius=4, padAngle=0.02).encode(
        theta=alt.Theta("count:Q"),
        color=alt.Color("sentiment:N", sort=order, scale=alt.Scale(domain=order, range=list(SENTIMENT_COLOURS.values())),
                        legend=alt.Legend(orient="right", title=None, labelFontSize=12, symbolType="circle")),
        tooltip=["sentiment", "count", "share"])
    return donut.properties(height=180).configure_view(strokeWidth=0).configure(background="transparent", font="Inter")


def _overview_actions(pipeline: VoxAuraPipeline, session: dict, insights: Insights | None) -> None:
    c1, c2 = st.columns(2)
    with c1:
        with st.popover("Export", icon=":material/picture_as_pdf:", width="stretch",
                        disabled=insights is None):
            if insights is not None:
                include = st.toggle("Include full transcript", value=False)
                pdf = None
                try:
                    from ..report import build_insights_pdf

                    pdf = build_insights_pdf(session, insights, pipeline.settings.chat_model,
                                             include_transcript=include)
                except ImportError:
                    st.error("PDF export needs ReportLab. Run `pip install -r requirements.txt` and restart.")
                except Exception as e:  # never let the export break the Overview page
                    st.error(f"Could not build the PDF: {friendly_error(e)}")
                if pdf is not None:
                    safe = re.sub(r"[^A-Za-z0-9]+", "_", insights.title or "insights").strip("_")[:50] or "insights"
                    st.download_button("Download PDF report", pdf, file_name=f"VoxAura_{safe}.pdf",
                                       mime="application/pdf", type="primary", icon=":material/download:",
                                       width="stretch")
    with c2:
        if st.button("Generate" if insights is None else "Regenerate", icon=":material/refresh:",
                     width="stretch", type="primary" if insights is None else "secondary",
                     key="regenerate"):
            with st.spinner("Analysing the transcript - this takes a few seconds..."):
                try:
                    pipeline.create_insights(session)
                    st.session_state["sessions"] = None  # the title may have changed
                except (InsightsParseError, Exception) as e:
                    notify("error", f"Could not generate insights: {friendly_error(e)}")
            st.rerun()


def render_insights_page(pipeline: VoxAuraPipeline, session: dict) -> None:
    insights = Insights.from_dict(session.get("insights"))
    page_header("Overview", session["title"], session_subtitle(session),
                actions=lambda: _overview_actions(pipeline, session, insights))

    m = session_metrics(session)
    stats = m["fillers"]
    kpis([
        ("clock", "Duration", m["duration"], SOURCE_LABELS.get(session["source_type"], "")),
        ("text", "Words", f'{m["words"]:,}', "in the clean transcript"),
        ("gauge", "Speaking rate", f'{m["wpm"]} wpm' if m["wpm"] else "—", "conversational: 120–160"),
        ("filler", "Filler words", str(stats.total_fillers), f"{stats.per_100_words} per 100 words"),
    ])

    left, right = st.columns([1.45, 1], gap="medium")
    with left:
        if insights is None:
            empty_state("spark", "No AI insights yet",
                        "Use <b>Generate</b> above to create a summary, tone analysis, key points and action items.")
        else:
            card("spark", "Executive summary", f"<p>{esc(insights.summary)}</p>")
            if insights.tone:
                card("smile", "Tone & speaking style", f"<p>{esc(insights.tone)}</p>")
            items = "".join(f"<li>{esc(p)}</li>" for p in insights.key_points) or "<li>None</li>"
            card("list", "Key points", f"<ul>{items}</ul>")
            st.caption(f"Generated by {pipeline.settings.chat_model} ({insights.method}). AI-generated content - "
                       "check important details against the transcript.")
    with right:
        if insights is not None:
            with st.container(key="panel-actions"):
                card_title("list", "Action items")
                if insights.action_items:
                    for i, item in enumerate(insights.action_items):
                        st.checkbox(item, key=f"action-{session['session_id']}-{i}")
                else:
                    st.caption("No action items were mentioned.")
            st.markdown("")
        with st.container(key="panel-fillers"):
            card_title("filler", "Filler words")
            if stats.counts:
                st.altair_chart(_filler_chart(stats.counts), width="stretch")
            else:
                st.caption("No filler words detected. Clear, fluent speech.")
        st.markdown("")
        with st.container(key="panel-sentiment"):
            card_title("smile", "Sentence sentiment")
            if m["sentiment"]:
                st.altair_chart(_sentiment_chart(m["sentiment"]), width="stretch")
                st.caption("AssemblyAI sentiment model, one label per sentence.")
            else:
                st.caption("Sentiment is available for audio transcribed with this version.")


# --------------------------------------------------------------------------- #
# Page: transcript
# --------------------------------------------------------------------------- #


def _highlight_query(escaped_html: str, query: str) -> str:
    if not query:
        return escaped_html
    pattern = re.compile(re.escape(html.escape(query)), re.IGNORECASE)
    return pattern.sub(lambda mt: f'<mark class="vx-hit">{mt.group(0)}</mark>', escaped_html)


def render_transcript_page(pipeline: VoxAuraPipeline, session: dict) -> None:
    def actions():
        st.download_button("Download .txt", session["text"], file_name=f"transcript_{session['session_id']}.txt",
                           icon=":material/download:", width="stretch")

    page_header("Transcript", session["title"], session_subtitle(session), actions=actions)
    stats = pipeline.cleaner.detect_fillers(session["raw_text"])

    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        view = st.segmented_control("View", ["Clean", "Timestamped", "Fillers highlighted"], default="Clean",
                                    key="transcript_view", label_visibility="collapsed") or "Clean"
    with right:
        query = st.text_input("Find", placeholder="Find in transcript", icon=":material/search:",
                              label_visibility="collapsed").strip()

    segments = session.get("segments") or [{"text": session["text"]}]
    with st.container(height=520, border=True):
        if view == "Fillers highlighted":
            legend = " ".join(f'<span><span class="vx-badge amber">{esc(k)}</span> × {v}</span>'
                              for k, v in stats.counts.items()) or "<span>No fillers detected.</span>"
            html_block(f'<div class="vx-legend"><span><b>{stats.total_fillers}</b> fillers found. They are '
                       f"removed from the clean version.</span></div><div class=\"vx-legend\">{legend}</div>")
            html_block(f'<div class="vx-prose">{pipeline.cleaner.highlight_html(session["raw_text"])}</div>')
        elif view == "Clean":
            text = _highlight_query(esc(session["text"]), query)
            html_block(f'<div class="vx-prose">{text}</div>')
        else:
            rows = [s for s in segments if query.lower() in s["text"].lower()]
            has_time = segments[0].get("start_ms") is not None
            if not has_time:
                st.caption("This transcript has no timestamps (pasted text or a .txt file).")
            if not rows:
                st.caption("No matching sentences.")
            html_block("".join(
                '<div class="vx-row">'
                + (f'<span class="vx-time">{format_timestamp(s.get("start_ms"))}</span>' if has_time else "<span></span>")
                + f'<span class="vx-sent {s.get("sentiment") or ""}" title="{esc((s.get("sentiment") or "no sentiment").lower())}"></span>'
                + f"<span>{_highlight_query(esc(s['text']), query)}</span></div>"
                for s in rows))

    with st.expander("How this transcript is indexed for retrieval (RAG)", icon=":material/schema:"):
        chunks = pipeline.chunks_for(session)
        st.caption(f"{len(chunks)} chunks · sentences grouped into ~{pipeline.settings.chunk_size}-character chunks "
                   f"with ~{pipeline.settings.chunk_overlap} characters of overlap, embedded with "
                   f"{pipeline.settings.embedding_model} and stored in this session's Chroma collection.")
        html_block("".join(
            f'<div class="vx-src"><div class="num">{c.index}</div><div><div class="meta"><span>{c.time_range}</span>'
            f"<span>{len(c.text)} chars</span></div><div class=\"txt\">{esc(c.text)}</div></div></div>"
            for c in chunks))


# --------------------------------------------------------------------------- #
# Page: Ask AI
# --------------------------------------------------------------------------- #


def _render_sources(sources: list[dict], answer: str, meta: dict | None = None) -> None:
    if not sources:
        return
    cited = cited_numbers(answer)
    if meta:
        badges = [f'<span class="vx-badge">{esc(meta.get("question_type", "").title())} question</span>',
                  f'<span class="vx-badge grey">Context: {esc(meta.get("strategy", ""))}</span>']
        if meta.get("rewritten"):
            badges.append(f'<span class="vx-badge grey">Searched: {esc(meta["search_query"])}</span>')
        html_block(f'<div class="vx-answer-meta">{"".join(badges)}</div>')
    n_cited = len(cited & set(range(1, len(sources) + 1)))
    with st.expander(f"Sources · {n_cited} cited of {len(sources)} used", icon=":material/format_quote:"):
        html_block("".join(
            f'<div class="vx-src{"" if i in cited else " dim"}"><div class="num">{i}</div><div>'
            f'<div class="meta"><span>{s["time_range"]}</span><span>chunk {s["chunk_index"]}</span>'
            f'<span>similarity {s["similarity"]:.2f}</span><span>{"cited" if i in cited else "not cited"}</span></div>'
            f'<div class="txt">{esc(s["text"])}</div></div></div>'
            for i, s in enumerate(sources, start=1)))


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
    page_header("Ask AI", session["title"],
                "Answers are grounded in the transcript. Analytical answers separate <b>evidence</b> from "
                "<b>interpretation</b>.")
    messages = st.session_state["messages"]

    # The question is read first so the suggestions below can already leave it out.
    # (st.chat_input is pinned to the bottom of the page wherever it is called.)
    question = st.chat_input("Ask anything about this recording...") or st.session_state.pop("pending_question", None)

    conversation = st.container()
    if not messages and not question:
        with conversation:
            empty_state("chat", "Start a conversation",
                        "Ask about facts, tone, intent or arguments. Pick a suggestion below or type your own question.")
            st.markdown("")

    asked = messages + ([{"role": "user", "content": question}] if question else [])
    st.caption("Suggested questions")
    with st.container(key="suggestions"):
        cols = st.columns(2)
        for i, q in enumerate(suggestions_for(session, asked)):
            cols[i % 2].button(q, key=f"suggest-{i}-{q}", icon=":material/north_east:", width="stretch",
                               on_click=_ask_suggestion, args=(q,))

    with conversation:
        for m in messages:
            with st.chat_message(m["role"], avatar=USER_AVATAR if m["role"] == "user" else BOT_AVATAR):
                st.markdown(escape_md(m["content"]))
                _render_sources(m.get("sources") or [], m["content"])
        if question:
            _answer(pipeline, session, question)
