import re

from voxaura.analysis.insights import Insights
from voxaura.report import _safe, build_insights_pdf

SESSION = {
    "title": "Phone launch", "source_type": "upload", "source_name": "launch.mp3", "created_at": "2026-10-25 12:00",
    "text": "Welcome to the launch. The phone has a three day battery.",
    "segments": [{"text": "Welcome to the launch.", "start_ms": 0, "end_ms": 30_000, "sentiment": "POSITIVE"},
                 {"text": "The phone has a three day battery 🔋.", "start_ms": 30_000, "end_ms": 95_000,
                  "sentiment": "NEUTRAL"}],
    "filler_stats": {"counts": {"um": 4, "like": 2}, "total_words": 210},
}
INSIGHTS = Insights(title="Phone launch event", summary="A launch with <demo> & “quotes”.",
                    key_points=["Battery lasts 3 days"], action_items=["Open pre-orders on Friday"],
                    tone="Upbeat.", suggested_questions=["How long does the battery last?"])


def page_count(pdf: bytes) -> int:
    return len(re.findall(rb"/Type /Page\b", pdf))


def test_builds_a_valid_pdf():
    pdf = build_insights_pdf(SESSION, INSIGHTS, "gpt-4o-mini")
    assert pdf.startswith(b"%PDF-") and pdf.rstrip().endswith(b"%%EOF")
    assert page_count(pdf) == 1


def test_transcript_appendix_adds_a_page():
    assert page_count(build_insights_pdf(SESSION, INSIGHTS, "gpt-4o-mini", include_transcript=True)) == 2


def test_handles_minimal_insights_and_no_sentiment():
    session = {**SESSION, "segments": [{"text": "Hi.", "start_ms": None, "end_ms": None}], "filler_stats": None}
    assert build_insights_pdf(session, Insights(summary="Short."), "model").startswith(b"%PDF-")


def test_text_is_escaped_and_made_printable():
    assert _safe("<b> & “ok”") == "&lt;b&gt; &amp; “ok”"  # curly quotes exist in the PDF font
    assert _safe("battery 🔋") == "battery ?"  # emoji does not
