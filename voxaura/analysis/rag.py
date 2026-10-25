"""Retrieval-Augmented Generation over a single transcript.

Pipeline
    1. Chunking   - consecutive sentences are grouped into ~chunk_size character
                    chunks with a small overlap; each chunk keeps its time range.
    2. Embedding  - each chunk is embedded (OpenAI embeddings via LangChain).
    3. Indexing   - vectors are stored in a Chroma collection, one per session.
    4. Retrieval  - the question is embedded and every chunk is scored by cosine
                    similarity.
    5. Routing    - the question is classified as "specific" or "analytical" and
                    the context is chosen accordingly (whole transcript, top-k,
                    or top-k plus a coverage sample).
    6. Generation - the LLM answers from that context, cites chunks as [n] and
                    keeps evidence separate from interpretation.

Only transcript chunks are ever stored in the vector database. Chat history is
kept separately and used for two things only: rewriting a follow-up question
into a standalone search query, and giving the LLM the conversation so far.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..ingestion.transcription import Segment, format_timestamp

# --------------------------------------------------------------------------- #
# 1. Chunking
# --------------------------------------------------------------------------- #


@dataclass
class Chunk:
    index: int
    text: str
    start_ms: int | None = None
    end_ms: int | None = None

    @property
    def time_range(self) -> str:
        if self.start_ms is None:
            return "no timestamp"
        return f"{format_timestamp(self.start_ms)}-{format_timestamp(self.end_ms)}"


def _split_long_segment(segment: Segment, max_chars: int) -> list[Segment]:
    """Split a single over-long sentence on word boundaries."""
    if len(segment.text) <= max_chars:
        return [segment]
    pieces, current = [], ""
    for word in segment.text.split():
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return [Segment(p, segment.start_ms, segment.end_ms) for p in pieces]


def chunk_segments(segments: list[Segment], chunk_size: int = 800, chunk_overlap: int = 150) -> list[Chunk]:
    """Group sentences into chunks without ever cutting a sentence in half.

    When a chunk is full, the last sentences (up to `chunk_overlap` characters)
    are carried into the next chunk, so an idea spanning a boundary is still
    retrievable from either side.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    pieces = [p for s in segments if s.text.strip() for p in _split_long_segment(s, chunk_size)]
    chunks: list[Chunk] = []
    current: list[Segment] = []
    length = 0

    def flush():
        chunks.append(Chunk(
            index=len(chunks),
            text=" ".join(s.text.strip() for s in current),
            start_ms=current[0].start_ms,
            end_ms=current[-1].end_ms,
        ))

    for piece in pieces:
        piece_len = len(piece.text) + 1
        if current and length + piece_len > chunk_size:
            flush()
            overlap, overlap_len = [], 0
            for s in reversed(current):
                if overlap_len + len(s.text) + 1 > chunk_overlap:
                    break
                overlap.insert(0, s)
                overlap_len += len(s.text) + 1
            current, length = overlap, overlap_len
        current.append(piece)
        length += piece_len
    if current:
        flush()
    return chunks


# --------------------------------------------------------------------------- #
# 2-4. Embedding, indexing and retrieval (Chroma)
# --------------------------------------------------------------------------- #


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float  # cosine similarity, higher = more relevant


class TranscriptIndex:
    """A Chroma collection holding the chunks of one session's transcript."""

    def __init__(self, session_id: str, embeddings: Embeddings, persist_dir: Path):
        from langchain_chroma import Chroma

        self.store = Chroma(
            collection_name=f"session_{session_id}",
            embedding_function=embeddings,
            persist_directory=str(persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return len(self.store.get(include=[])["ids"])

    def build(self, chunks: list[Chunk]) -> bool:
        """Embed and store chunks. Returns False if an up-to-date index already exists."""
        existing = self.store.get(include=[])["ids"]
        if len(existing) == len(chunks):
            return False  # reloading a past session - no need to pay for embeddings again
        if existing:
            self.store.delete(ids=existing)
        self.store.add_texts(
            texts=[c.text for c in chunks],
            metadatas=[
                # Chroma metadata values cannot be None, so -1 means "no timestamp"
                {"chunk_index": c.index, "start_ms": c.start_ms if c.start_ms is not None else -1,
                 "end_ms": c.end_ms if c.end_ms is not None else -1}
                for c in chunks
            ],
            ids=[f"chunk-{c.index}" for c in chunks],
        )
        return True

    def search(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        k = min(k, self.count())
        if k == 0:
            return []
        results = self.store.similarity_search_with_score(query, k=k)
        retrieved = []
        for doc, distance in results:
            meta = doc.metadata
            chunk = Chunk(
                index=int(meta["chunk_index"]),
                text=doc.page_content,
                start_ms=None if meta["start_ms"] < 0 else int(meta["start_ms"]),
                end_ms=None if meta["end_ms"] < 0 else int(meta["end_ms"]),
            )
            retrieved.append(RetrievedChunk(chunk, round(1 - distance, 3)))
        return retrieved

    def delete(self) -> None:
        self.store.delete_collection()


# --------------------------------------------------------------------------- #
# 5. Question routing and context selection
# --------------------------------------------------------------------------- #

# Questions about the recording as a whole. These do not resemble any single
# passage, so plain top-k similarity search retrieves them poorly: "What is the
# main topic?" scored only ~0.18 against every chunk of a real speech.
_ANALYTICAL_RE = re.compile(
    r"\b(tone|sentiment|mood|emotion\w*|feel(s|ing)?|attitude|style|speak(er|ing)?|deliver\w*|rhetoric\w*|"
    r"persua\w*|convinc\w*|intent\w*|purpose|goal|motiv\w*|argu\w*|claims?|reason\w*|topics?|themes?|"
    r"main|overall|summar\w*|gist|takeaways?|message|opinions?|bias\w*|audience|stance|"
    r"position|confiden\w*|formal|informal|tactic\w*|strateg\w*|strongest|weakest|key points?|"
    r"impression|credib\w*|honest\w*|positive|negative|optimis\w*|pessimis\w*|why)\b",
    re.IGNORECASE,
)
# "What is this recording about?" is a whole-recording question, but
# "What does he say about taxes?" is a specific one.
_ABOUT_RE = re.compile(r"\b(is|was|are|were)\s+(this|it|the\s+\w+|they)(\s+\w+)?\s+about\s*\??$", re.IGNORECASE)


def classify_question(question: str) -> str:
    """Return "analytical" for whole-recording / interpretive questions, else "specific"."""
    q = question.strip()
    return "analytical" if _ANALYTICAL_RE.search(q) or _ABOUT_RE.search(q) else "specific"


def coverage_sample(chunks: list[RetrievedChunk], n: int) -> list[RetrievedChunk]:
    """Pick n chunks spread evenly from the start to the end of the recording."""
    ordered = sorted(chunks, key=lambda r: r.chunk.index)
    if len(ordered) <= n:
        return ordered
    step = (len(ordered) - 1) / (n - 1) if n > 1 else 0
    return [ordered[round(i * step)] for i in range(n)]


@dataclass
class ContextSelection:
    chunks: list[RetrievedChunk]  # in the order they are shown to the LLM
    strategy: str  # human-readable description, shown in the UI


def select_context(ranked: list[RetrievedChunk], question_type: str, k: int,
                   full_context_chars: int, coverage: int = 6) -> ContextSelection:
    """Choose which chunks the LLM sees.

    `ranked` holds every chunk, most similar first.
      * Short transcript -> the whole transcript, in order (nothing can be missed).
      * Long transcript, analytical question -> top-k similar chunks plus an even
        sample across the recording, so the answer reflects the whole of it.
      * Long transcript, specific question -> top-k similar chunks (classic RAG).
    """
    if sum(len(r.chunk.text) for r in ranked) <= full_context_chars:
        return ContextSelection(sorted(ranked, key=lambda r: r.chunk.index), "full transcript")
    if question_type == "analytical":
        chosen = {r.chunk.index: r for r in ranked[:k] + coverage_sample(ranked, coverage)}
        return ContextSelection(sorted(chosen.values(), key=lambda r: r.chunk.index),
                                f"top-{k} semantic search + coverage sample across the recording")
    return ContextSelection(ranked[:k], f"top-{k} semantic search")


# --------------------------------------------------------------------------- #
# 6. Generation
# --------------------------------------------------------------------------- #

CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's latest question as a standalone question that can be understood "
     "without the conversation. Resolve pronouns like 'he', 'it' or 'that'. "
     "If it is already standalone, return it unchanged. Return only the question."),
    MessagesPlaceholder("history"),
    ("human", "Latest question: {question}"),
])

ANSWER_SYSTEM = """You are VoxAura, an analyst who answers questions about one audio recording using its transcript.

You are given two kinds of input:
- TRANSCRIPT EXCERPTS, numbered [1], [2], ... with time ranges. This is your evidence.
- RECORDING FACTS, measured by the app (length, speaking rate, filler words, and sentence-level sentiment from a speech model when available).

How to answer:
1. Specific questions (who, what, when, how many): answer directly in 1-3 sentences and cite the excerpts you used, e.g. [2] or [1][3].
2. Analytical questions (tone, sentiment, speaking style, intent, persuasion, key arguments, main topics, overall message): give a real analysis, not a list of quotes. Use exactly these sections:
   **Answer** - a 1-3 sentence conclusion.
   **Evidence** - 2-5 bullets, each a short quote or precise paraphrase with its citation [n], or a recording fact.
   **Interpretation** - what the evidence suggests and why, in hedged language ("suggests", "likely"), ending with your confidence (low / medium / high).
3. Every answer that describes the recording must cite at least one excerpt. Keep evidence and interpretation separate. Never present an interpretation as a fact. Never invent quotes, speakers, names, numbers or events that are not in the excerpts. General knowledge may be used only to explain a concept (for example, what a rhetorical device is), never to add facts about the recording.
4. You cannot hear the audio. Tone and emotion are inferred from the wording and, if provided, the sentiment model - say so when it matters.
5. Broad questions about the recording (its topic, tone, purpose, arguments) can always be answered from the excerpts. Only when the excerpts contain nothing relevant to a specific question, say "The recording doesn't seem to mention that." and briefly say what it does cover.
6. Questions about this conversation itself (for example "what was my first question?") may be answered from the chat history.

RECORDING FACTS:
{facts}

TRANSCRIPT EXCERPTS ({strategy}):
{context}"""

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def cited_numbers(answer: str) -> set[int]:
    """Citation numbers used in an answer: handles [1], [2][3] and [1, 2]."""
    return {int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer) for n in group.split(",")}


@dataclass
class RagAnswer:
    answer: str
    sources: list[RetrievedChunk]
    search_query: str
    question_type: str = "specific"
    strategy: str = ""

    @property
    def cited_numbers(self) -> set[int]:
        return cited_numbers(self.answer)


def format_context(retrieved: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({r.chunk.time_range}) {r.chunk.text}" for i, r in enumerate(retrieved, start=1)
    )


def to_langchain_history(messages: list[dict], max_turns: int) -> list:
    """Convert stored {"role", "content"} dicts into LangChain messages (most recent turns only)."""
    recent = messages[-max_turns * 2:] if max_turns else []
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in recent
    ]


class TranscriptChat:
    def __init__(self, llm: BaseChatModel, index: TranscriptIndex, k: int = 4, history_turns: int = 4,
                 full_context_chars: int = 12_000, recording_facts: str = ""):
        self.index = index
        self.k = k
        self.history_turns = history_turns
        self.full_context_chars = full_context_chars
        self.recording_facts = recording_facts or "(none available)"
        self.condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()
        self.answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    def make_search_query(self, question: str, history: list) -> str:
        # Follow-ups like "why did he say that?" retrieve poorly on their own,
        # so they are rewritten first. The first question needs no rewriting.
        if not history:
            return question
        return self.condense_chain.invoke({"history": history, "question": question}).strip() or question

    def ask(self, question: str, chat_history: list[dict] | None = None) -> RagAnswer:
        history = to_langchain_history(chat_history or [], self.history_turns)
        search_query = self.make_search_query(question, history)
        question_type = classify_question(search_query)
        # Every chunk is scored (a transcript has at most a few hundred), then
        # select_context decides which of them the LLM sees.
        ranked = self.index.search(search_query, k=self.index.count())
        selection = select_context(ranked, question_type, self.k, self.full_context_chars)
        answer = self.answer_chain.invoke({
            "facts": self.recording_facts,
            "strategy": selection.strategy,
            "context": format_context(selection.chunks) or "(the transcript is empty)",
            "history": history,
            "question": question,
        })
        return RagAnswer(answer.strip(), selection.chunks, search_query, question_type, selection.strategy)
