"""VoxAuraAI - Streamlit entry point.

Run with:  streamlit run app.py
"""

import logging

import streamlit as st

st.set_page_config(page_title="VoxAuraAI", page_icon="🔉", layout="wide")

# Draw something immediately: the AI libraries below take a few seconds to import
# on a cold start, which otherwise shows as a blank white page.
_loading = st.empty()
_loading.markdown("### 🔉 VoxAuraAI\nLoading the AI components...")

from voxaura import ui  # noqa: E402
from voxaura.config import load_settings  # noqa: E402
from voxaura.database import Database, DatabaseError  # noqa: E402
from voxaura.pipeline import VoxAuraPipeline  # noqa: E402
from voxaura.text_cleaning import TextCleaner  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# Chroma 0.6 logs harmless errors from its (disabled) usage telemetry.
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)


@st.cache_resource(show_spinner="Connecting to the database and AI services...")
def get_pipeline() -> VoxAuraPipeline:
    """Create the shared objects once per server process (not on every rerun)."""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    settings = load_settings()
    return VoxAuraPipeline(
        settings=settings,
        db=Database.from_settings(settings),
        cleaner=TextCleaner.from_config_dir(settings.config_dir),
        embeddings=OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key),
        llm=ChatOpenAI(model=settings.chat_model, temperature=0.2, api_key=settings.openai_api_key),
    )


def main() -> None:
    ui.init_state()
    ui.load_css()

    missing = load_settings().missing_keys()
    if missing:
        _loading.empty()
        st.error(f"Missing configuration: {', '.join(missing)}. Copy `.env.example` to `.env` and fill it in.")
        st.stop()
    try:
        pipeline = get_pipeline()
    except DatabaseError as e:
        _loading.empty()
        st.error(f"{e}\n\nCheck the DB_* settings in `.env`, or remove DB_HOST to use local SQLite instead.")
        st.stop()
    _loading.empty()

    ui.render_sidebar(pipeline)
    ui.render_header()
    page = ui.render_nav()

    session = st.session_state["session"]
    if page == ui.INPUT_PAGE:
        ui.render_input_page(pipeline)
    elif session is None:
        ui.empty_state("🎧", "No session open",
                       "Add a recording on the <b>Add audio</b> page, or open a past session from the sidebar.")
    elif page == ui.TRANSCRIPT_PAGE:
        ui.render_transcript_page(pipeline, session)
    elif page == ui.INSIGHTS_PAGE:
        ui.render_insights_page(pipeline, session)
    else:
        ui.render_chat_page(pipeline, session)


main()
