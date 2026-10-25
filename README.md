# 🔉 VoxAuraAI — AI-Powered Audio Understanding and Conversation

VoxAuraAI turns spoken audio into something you can read, analyse and talk to.
It transcribes a recording, cleans the transcript, generates AI insights, and
lets you ask questions that are answered **from the recording itself** using
Retrieval-Augmented Generation (RAG), with timestamped source references.

*MSc Artificial Intelligence — final-year project.*

## Features

| Area | What it does |
|---|---|
| **Audio input** | Upload a file (mp3, wav, m4a, ogg, flac, webm, mp4), record live from the microphone, paste a YouTube link, or supply a transcript (.txt, .md, .srt, .vtt or pasted text). All of these go through one shared pipeline. |
| **Speech-to-text** | AssemblyAI transcription with sentence timestamps and sentence-level sentiment. |
| **Transcript cleaning** | Filler detection and removal. Hesitations (`um`, `uh`) are always fillers; contextual fillers (`like`, `you know`, `I mean`) only count when set off by commas, so "I like pizza" is untouched. Chat shorthand is expanded (`idk` → `I don't know`), and acronyms are explained on first use (`AI` → `AI (artificial intelligence)`). Dotted acronyms such as `U.S.` are never treated as fillers or shorthand. The word lists are in `config/*.json`. |
| **Speech analysis** | Duration, speaking rate (words per minute), fillers per 100 words, a filler chart and the sentence-sentiment breakdown. |
| **AI insights** | Title, summary, tone, key points, action items and suggested questions, generated automatically after every upload or recording. Long transcripts are handled with map-reduce summarisation. |
| **RAG chat** | Specific questions use top-k semantic retrieval. Broad or analytical questions (tone, intent, arguments, main topic) get the whole transcript, or for long recordings a sample covering all of it. Analytical answers are split into **Answer / Evidence / Interpretation**, so cited evidence is kept apart from the model's inferences. Sources show timestamps and similarity scores, and follow-up questions are rewritten using the chat history. |
| **History** | Sessions, insights and chat (including sources) are saved to MySQL, or to SQLite when no MySQL server is configured, and can be reopened from the sidebar. |

## Quick start

**Requirements:** Python 3.10+, an [AssemblyAI](https://www.assemblyai.com/) API key and an [OpenAI](https://platform.openai.com/) API key.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
copy .env.example .env          # Windows  (cp on macOS / Linux)
#    then edit .env and add ASSEMBLYAI_API_KEY and OPENAI_API_KEY

# 4. Run
streamlit run app.py
```

The app opens at <http://localhost:8501>.

### Database

* **No setup needed by default.** If `DB_HOST` is empty, a SQLite file is created at `data/voxaura.db`.
* **To use MySQL,** fill in `DB_HOST`, `DB_USER`, `DB_PASSWORD` and `DB_DATABASE`. The tables are created on first run. If you already have the tables from the earlier version of VoxAuraAI, the new columns are added automatically, so old sessions remain available. The database user needs `CREATE` and `ALTER` permissions for this first run.

### Configuration reference

All settings are environment variables (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `ASSEMBLYAI_API_KEY` | — | Speech-to-text (required) |
| `OPENAI_API_KEY` | — | LLM and embeddings (required) |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | LLM for answers and insights |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model for RAG |
| `CHUNK_SIZE` | `800` | Target characters per chunk |
| `CHUNK_OVERLAP` | `150` | Characters of overlap between chunks |
| `RETRIEVAL_K` | `4` | Chunks retrieved per question |
| `FULL_CONTEXT_CHARS` | `12000` | Transcripts up to this length are given to the LLM in full |
| `DB_HOST`, `DB_USER`, `DB_PASSWORD`, `DB_DATABASE` | empty | MySQL connection (empty = SQLite) |
| `DATA_DIR` | `data` | Where SQLite, the vector store and audio files are kept |

## Using the app

Use the navigation bar under the header to move between pages. Past sessions are listed in the sidebar.

1. **🎧 Add audio**: choose an input method. A progress bar and a step log show each stage of the pipeline, and when it finishes the app opens the insights page.
   *Live recording:* the browser asks for microphone permission. Microphones only work on `localhost` or HTTPS. Recordings that are shorter than one second or silent are rejected with an explanation.
2. **📝 Transcript**: switch between the clean text, a timestamped view (with sentiment dots) and the original with fillers highlighted, and search within the transcript. Expand *How this transcript is indexed* to see the chunks stored in the vector database.
3. **✨ Insights**: speech statistics, the summary, tone, key points, action items and charts. Use *Regenerate* to run the analysis again.
4. **💬 Ask AI**: click a suggested question or type your own. The suggestions stay visible after each answer. Open *📎 Sources* under an answer to see the question type, the context strategy, and the chunks that were used and cited.

## Project structure

```
VoxAuraAI/
├── app.py                  # Streamlit entry point (tabs + wiring)
├── voxaura/
│   ├── config.py           # Settings from environment variables
│   ├── audio_input.py      # Upload / recording / YouTube → local audio file
│   ├── transcription.py    # AssemblyAI client (upload, poll, sentences)
│   ├── text_cleaning.py    # Filler detection/removal, abbreviation expansion
│   ├── transcript_files.py # Validating/parsing .txt, .md, .srt, .vtt transcripts
│   ├── rag.py              # Chunking, Chroma index, question routing, grounded answering
│   ├── insights.py         # Summary / key points / action items (map-reduce)
│   ├── database.py         # MySQL or SQLite persistence
│   ├── pipeline.py         # Shared processing pipeline (no UI code)
│   ├── ui.py               # Streamlit components for each page
│   └── styles.css          # Visual design (typography, colours, cards)
├── .streamlit/config.toml  # Streamlit theme
├── config/
│   ├── fillers.json        # Filler words to detect and remove
│   └── abbreviations.json  # Abbreviations to expand
├── tests/                  # pytest suite (runs offline, no API keys)
├── docs/ARCHITECTURE.md    # Architecture and RAG workflow (for evaluation)
└── legacy/                 # The original separate apps, kept for reference
```

## Running the tests

```bash
pytest
```

The tests need **no API keys or network access**. They use a small
deterministic bag-of-words embedding and a fake chat model, so the real
chunking, Chroma indexing, retrieval, prompt construction, database and
pipeline code all run. The tests cover:

* fillers: hesitations, contextual fillers ("like", "you know") versus the same words used with their real meaning, and "U.S."/"us" never counted as fillers
* abbreviations: chat shorthand, acronyms explained once, and "ATM"/"Vitamin K" left alone
* transcript uploads: encodings (UTF-8 BOM, UTF-16, cp1252), SRT/VTT timestamps, and validation errors
* recordings: valid, too short, silent and corrupt files
* chunking, retrieval, question routing (full transcript / top-k / coverage sample), and keeping chat history out of the retrieved context
* insights parsing, and choosing between a single call and map-reduce
* database round trips and upgrading tables from the old schema
* the full pipeline (audio, recording and text) through to automatic insights and a cited answer
* **UI tests** (`tests/test_app.py`) that drive the real Streamlit app with `AppTest`: the whole text-upload flow, clicking suggested questions (they answer and stay visible), switching sessions, and error messages

## Limitations

* Transcription and LLM calls need an internet connection and paid API keys.
* Tone and emotion are inferred from the words and AssemblyAI's text-based sentiment model. The app does not analyse pitch or voice, and answers say so when it matters.
* Sentiment analysis and filler detection (`disfluencies`) are English-only features of AssemblyAI.
* Contextual fillers are detected from punctuation, so they depend on AssemblyAI placing the commas. "I like, really liked it" (no comma before "like") is not detected.
* Question routing uses keywords. An unusual analytical question may be treated as specific; for short recordings this makes no difference, because the full transcript is used either way.
* Suggested questions for very short recordings can ask about things the recording doesn't say. The chat then replies that it isn't mentioned.
* One recording per session. Questions cannot span several recordings.
* Very long recordings (more than about 1 hour) may exceed the MySQL `TEXT` limit of tables created by the earlier version.
* YouTube downloads depend on `yt-dlp` and may break when YouTube changes. Keep `yt-dlp` updated.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design and the RAG workflow.
