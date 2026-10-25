"""LLM-generated insights: title, summary, key points and action items.

Short transcripts are sent to the LLM in one call. Long transcripts use a
map-reduce approach: each section is condensed into notes (map, run in
parallel), then the notes are combined into the final insights (reduce). This
keeps every request within the model's context window.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .rag import Chunk

MAX_DIRECT_CHARS = 24_000  # roughly 6k tokens

MAP_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are taking notes on one section of a longer audio transcript. List the main "
     "points, any decisions, and any tasks or commitments (with who is responsible, if "
     "stated). Be concise and use bullet points. Do not invent details."),
    ("human", "Transcript section:\n{text}"),
])

INSIGHTS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You analyse transcripts of audio recordings. Using only the provided {source_kind}, "
     "return a JSON object with exactly these keys:\n"
     '  "title": a short descriptive title (max 8 words)\n'
     '  "summary": a 3-5 sentence summary\n'
     '  "key_points": a list of 3-7 of the most important points\n'
     '  "action_items": a list of concrete tasks, follow-ups or recommendations mentioned '
     "or clearly implied, each as one sentence (include the owner or deadline if stated). "
     "Use an empty list if there are none.\n"
     '  "tone": one sentence describing the overall tone and speaking style, based on the wording\n'
     '  "suggested_questions": 4 short questions a listener might ask about THIS recording that '
     "it can actually answer - mix questions about its arguments, intent or tone with "
     "questions about specific details it mentions\n"
     "Do not invent details. For a very short or casual recording keep every field brief.\n"
     "Return only the JSON object, no other text."),
    ("human", "{text}"),
])


@dataclass
class Insights:
    title: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    tone: str = ""
    suggested_questions: list[str] = field(default_factory=list)
    method: str = "direct"  # "direct" or "map-reduce"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "Insights | None":
        if not data:
            return None
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})

    def to_markdown(self) -> str:
        lines = [f"# {self.title or 'Insights'}", "", "## Summary", self.summary, ""]
        if self.tone:
            lines += ["## Tone", self.tone, ""]
        lines += ["## Key points"]
        lines += [f"- {p}" for p in self.key_points] or ["- None"]
        lines += ["", "## Action items"]
        lines += [f"- [ ] {a}" for a in self.action_items] or ["- None"]
        return "\n".join(lines)


class InsightsParseError(ValueError):
    pass


def parse_insights_json(raw: str) -> Insights:
    """Parse the LLM's JSON reply, tolerating ```json fences or extra text around it."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise InsightsParseError("The model did not return JSON.")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise InsightsParseError(f"The model returned invalid JSON: {e}") from e

    def as_list(value) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(v).strip() for v in value or [] if str(v).strip()]

    return Insights(
        title=str(data.get("title", "")).strip(),
        summary=str(data.get("summary", "")).strip(),
        key_points=as_list(data.get("key_points")),
        action_items=as_list(data.get("action_items")),
        tone=str(data.get("tone", "")).strip(),
        suggested_questions=[q if q.endswith("?") else q + "?"
                             for q in as_list(data.get("suggested_questions"))][:5],
    )


def group_chunks(chunks: list[Chunk], max_chars: int) -> list[str]:
    """Join consecutive chunks into sections of at most ~max_chars characters."""
    sections, current = [], ""
    for chunk in chunks:
        if current and len(current) + len(chunk.text) > max_chars:
            sections.append(current)
            current = ""
        current = f"{current}\n{chunk.text}".strip()
    if current:
        sections.append(current)
    return sections


def generate_insights(llm: BaseChatModel, transcript: str, chunks: list[Chunk],
                      max_direct_chars: int = MAX_DIRECT_CHARS) -> Insights:
    insights_chain = INSIGHTS_PROMPT | llm | StrOutputParser()

    if len(transcript) <= max_direct_chars:
        raw = insights_chain.invoke({"source_kind": "transcript", "text": transcript})
        method = "direct"
    else:
        map_chain = MAP_PROMPT | llm | StrOutputParser()
        sections = group_chunks(chunks, max_direct_chars // 2)
        notes = map_chain.batch([{"text": s} for s in sections])  # map (parallel)
        combined = "\n\n".join(f"Section {i}:\n{n}" for i, n in enumerate(notes, start=1))
        raw = insights_chain.invoke({"source_kind": "notes from each section of a long transcript",
                                     "text": combined})  # reduce
        method = "map-reduce"

    insights = parse_insights_json(raw)
    insights.method = method
    return insights
