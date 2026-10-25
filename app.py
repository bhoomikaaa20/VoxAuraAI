"""VoxAuraAI - Streamlit entry point.

Run with:  streamlit run app.py
"""

import logging

import streamlit as st

st.set_page_config(page_title="VoxAura · Audio intelligence", page_icon="🔉", layout="wide",
                   initial_sidebar_state="auto")  # open on desktop, collapsed on phones

# Draw something immediately: the AI libraries below take a few seconds to import
# on a cold start, which otherwise shows as a blank white page.
_loading = st.empty()
_loading.markdown("#### VoxAura\nLoading the AI components...")

from voxaura.analysis.text_cleaning import TextCleaner  # noqa: E402
from voxaura.config import Settings, load_settings  # noqa: E402
from voxaura.database import Database, DatabaseError  # noqa: E402
from voxaura.pipeline import VoxAuraPipeline  # noqa: E402
from voxaura.ui import pages  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# Chroma 0.6 logs harmless errors from its (disabled) usage telemetry.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


@st.cache_resource(show_spinner="Connecting to the database and AI services...", max_entries=1)
def get_pipeline(settings: Settings) -> VoxAuraPipeline:
    """Create the shared objects once per server process (not on every rerun).

    The cache is keyed on the settings, so editing .env (for example fixing an
    API key) builds a fresh pipeline on the next refresh instead of reusing the
    old key until the server is restarted.
    """
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    return VoxAuraPipeline(
        settings=settings,
        db=Database.from_settings(settings),
        cleaner=TextCleaner.from_config_dir(settings.config_dir),
        embeddings=OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key),
        llm=ChatOpenAI(model=settings.chat_model, temperature=0.2, api_key=settings.openai_api_key),
    )


def main() -> None:
    pages.init_state()
    pages.load_css()

    settings = load_settings()
    missing = settings.missing_keys()
    if missing:
        _loading.empty()
        st.error(f"Missing configuration: {', '.join(missing)}. Copy `.env.example` to `.env` and fill it in.")
        st.stop()
    try:
        pipeline = get_pipeline(settings)
    except DatabaseError as e:
        _loading.empty()
        st.error(f"{e}\n\nCheck the DB_* settings in `.env`, or remove DB_HOST to use local SQLite instead.")
        st.stop()
    _loading.empty()

    pages.sync_url(pipeline)
    pages.render_sidebar(pipeline)
    pages.show_notices()

    page = st.session_state["page"]
    session = st.session_state["session"]
    if page == pages.INPUT_PAGE:
        pages.render_input_page(pipeline)
    elif session is None:
        pages.page_header("Workspace", "No session open")
        pages.empty_state("wave", "No session open",
                       "Start a <b>New analysis</b>, or open a recent session from the sidebar.")
    elif page == pages.TRANSCRIPT_PAGE:
        pages.render_transcript_page(pipeline, session)
    elif page == pages.INSIGHTS_PAGE:
        pages.render_insights_page(pipeline, session)
    else:
        pages.render_chat_page(pipeline, session)


main()
