import pytest

from voxaura.insights import InsightsParseError, generate_insights, parse_insights_json
from voxaura.rag import Chunk
from tests.conftest import RecordingChatModel

JSON_REPLY = """```json
{"title": "Planning meeting", "summary": "The team planned next year.",
 "key_points": ["Budget up 20%", "Website redesign"], "action_items": ["Priya: redesign site by March"]}
```"""


def test_parses_json_inside_code_fences():
    insights = parse_insights_json(JSON_REPLY)
    assert insights.title == "Planning meeting"
    assert insights.key_points == ["Budget up 20%", "Website redesign"]
    assert insights.action_items == ["Priya: redesign site by March"]


def test_missing_fields_default_to_empty():
    insights = parse_insights_json('{"summary": "Short."}')
    assert insights.summary == "Short."
    assert insights.key_points == [] and insights.action_items == []


def test_invalid_reply_raises():
    with pytest.raises(InsightsParseError):
        parse_insights_json("Sorry, I can't help with that.")


def test_short_transcript_uses_single_call():
    llm = RecordingChatModel(responses=[JSON_REPLY])
    insights = generate_insights(llm, "A short transcript.", [Chunk(0, "A short transcript.")])
    assert insights.method == "direct"
    assert len(llm.calls) == 1


def test_long_transcript_uses_map_reduce():
    chunks = [Chunk(i, f"Section {i} " + "words " * 40) for i in range(10)]
    transcript = " ".join(c.text for c in chunks)
    llm = RecordingChatModel(responses=[JSON_REPLY])  # the fake repeats this reply for every call
    insights = generate_insights(llm, transcript, chunks, max_direct_chars=1000)

    assert insights.method == "map-reduce"
    sections = len(llm.calls) - 1  # map calls + one reduce call
    assert sections >= 2
    assert "notes from each section" in llm.calls[-1][0].content
