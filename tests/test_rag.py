import pytest

from voxaura.rag import (Chunk, RetrievedChunk, TranscriptChat, TranscriptIndex, cited_numbers,
                         classify_question, select_context)
from tests.conftest import RecordingChatModel

CHUNKS = [
    Chunk(0, "Welcome everyone to the quarterly planning meeting.", 0, 5000),
    Chunk(1, "The marketing budget will increase by twenty percent next year.", 5000, 12000),
    Chunk(2, "Priya will deliver the website redesign before the March deadline.", 12000, 20000),
    Chunk(3, "Lunch will be pizza and salad in the kitchen.", 20000, 25000),
]


@pytest.fixture
def index(tmp_path, embeddings):
    idx = TranscriptIndex("test", embeddings, tmp_path)
    idx.build(CHUNKS)
    return idx


# ------------------------------ retrieval ------------------------------ #

def test_retrieves_most_relevant_chunk(index):
    results = index.search("How much will the marketing budget increase?", k=2)
    assert results[0].chunk.index == 1
    assert results[0].similarity > results[1].similarity
    assert results[0].chunk.start_ms == 5000  # metadata survives the round trip


def test_rebuild_is_skipped_when_index_is_current(index):
    assert index.build(CHUNKS) is False
    assert index.count() == len(CHUNKS)


def test_k_larger_than_index_is_safe(index):
    assert len(index.search("meeting", k=50)) == len(CHUNKS)


# ------------------------------- routing ------------------------------- #

@pytest.mark.parametrize("question, expected", [
    ("What is the main topic of this recording?", "analytical"),
    ("How would you describe the speaker's tone?", "analytical"),
    ("What is the speaker trying to persuade the audience of?", "analytical"),
    ("What are the key arguments?", "analytical"),
    ("What is this recording about?", "analytical"),
    ("What does he say about taxes?", "specific"),
    ("When is the website deadline?", "specific"),
    ("How much will the budget increase?", "specific"),
])
def test_classify_question(question, expected):
    assert classify_question(question) == expected


def ranked(n, size=100):
    """n chunks, ranked so that the *last* chunks are the most similar."""
    return [RetrievedChunk(Chunk(i, "x" * size), round(1 - i / n, 2)) for i in reversed(range(n))]


def test_short_transcript_uses_full_transcript_in_order():
    selection = select_context(ranked(5), "specific", k=2, full_context_chars=10_000)
    assert selection.strategy == "full transcript"
    assert [r.chunk.index for r in selection.chunks] == [0, 1, 2, 3, 4]


def test_long_transcript_specific_question_uses_top_k():
    selection = select_context(ranked(50), "specific", k=3, full_context_chars=1_000)
    assert [r.chunk.index for r in selection.chunks] == [49, 48, 47]


def test_long_transcript_analytical_question_adds_coverage_sample():
    selection = select_context(ranked(50), "analytical", k=3, full_context_chars=1_000, coverage=5)
    indexes = [r.chunk.index for r in selection.chunks]
    assert {47, 48, 49} <= set(indexes)  # the most similar chunks
    assert 0 in indexes and 12 in indexes  # plus chunks spread across the recording
    assert indexes == sorted(indexes)  # shown in recording order


def test_cited_numbers_formats():
    assert cited_numbers("Yes [1]. Also [2][3] and [4, 5].") == {1, 2, 3, 4, 5}


# ------------------------------ generation ----------------------------- #

def test_specific_question_on_long_transcript_uses_top_k_context(index):
    llm = RecordingChatModel(responses=["The budget increases by twenty percent [1]."])
    chat = TranscriptChat(llm, index, k=2, full_context_chars=0)
    result = chat.ask("How much will the marketing budget increase?")

    assert len(llm.calls) == 1  # no question rewriting without history
    system_prompt = llm.calls[0][0].content
    assert "[1] (00:05-00:12) The marketing budget" in system_prompt
    assert "Lunch will be pizza" not in system_prompt
    assert result.sources[0].chunk.index == 1
    assert result.question_type == "specific"
    assert result.cited_numbers == {1}


def test_broad_question_gets_whole_short_transcript_and_recording_facts(index):
    """Regression: "What is the main topic?" used to retrieve 2 low-similarity chunks and fail."""
    llm = RecordingChatModel(responses=["**Answer** A planning meeting [1][2]."])
    chat = TranscriptChat(llm, index, k=2, recording_facts="- Speaking rate: about 150 words per minute")
    result = chat.ask("What is the main topic of this recording?")

    system_prompt = llm.calls[0][0].content
    assert result.question_type == "analytical"
    assert result.strategy == "full transcript"
    assert len(result.sources) == len(CHUNKS)
    assert all(c.text in system_prompt for c in CHUNKS)
    assert "about 150 words per minute" in system_prompt
    assert "**Evidence**" in system_prompt and "**Interpretation**" in system_prompt


def test_follow_up_is_rewritten_and_history_kept_out_of_context(index):
    history = [
        {"role": "user", "content": "Who is working on the website?"},
        {"role": "assistant", "content": "Priya is redesigning the website [1]."},
    ]
    llm = RecordingChatModel(responses=[
        "When is the website redesign deadline?",  # 1st call: standalone question
        "Before the March deadline [1].",  # 2nd call: answer
    ])
    result = TranscriptChat(llm, index, k=1, full_context_chars=0).ask("When is it due?", history)

    assert len(llm.calls) == 2
    assert result.search_query == "When is the website redesign deadline?"
    assert result.sources[0].chunk.index == 2

    answer_messages = llm.calls[1]
    system_prompt = answer_messages[0].content
    # retrieved transcript goes in the system prompt; chat history stays as separate turns
    assert "Priya will deliver" in system_prompt
    assert "Priya is redesigning" not in system_prompt
    assert [m.type for m in answer_messages] == ["system", "human", "ai", "human"]
