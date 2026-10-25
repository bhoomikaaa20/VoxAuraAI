"""PDF export of a session's AI insights, built with ReportLab."""

from __future__ import annotations

import html
import io
import unicodedata
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from .analysis.insights import Insights
from .pipeline import session_metrics

PRIMARY = colors.HexColor("#6D28D9")
INK = colors.HexColor("#1F1B2E")
MUTED = colors.HexColor("#6B6680")
SOFT = colors.HexColor("#F3EFFC")
BORDER = colors.HexColor("#E4DEF2")
SENTIMENT_COLOURS = {"POSITIVE": "#16A34A", "NEUTRAL": "#8B8799", "NEGATIVE": "#DC2626"}

STYLES = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=INK),
    "meta": ParagraphStyle("meta", fontName="Helvetica", fontSize=9, leading=13, textColor=MUTED),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=PRIMARY,
                         spaceBefore=12, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.5, leading=15.5, textColor=INK,
                           alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.5, leading=12, textColor=MUTED),
    "kpi_label": ParagraphStyle("kpi_label", fontName="Helvetica", fontSize=7.5, leading=10, textColor=MUTED),
    "kpi_value": ParagraphStyle("kpi_value", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=INK),
}


def _safe(text: str) -> str:
    """Escape for ReportLab markup and replace characters the built-in fonts cannot draw."""
    chars = []
    for ch in str(text or ""):
        try:
            ch.encode("cp1252")
            chars.append(ch)
        except UnicodeEncodeError:
            chars.append(unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode() or "?")
    return html.escape("".join(chars))


def _p(text: str, style: str = "body") -> Paragraph:
    return Paragraph(_safe(text), STYLES[style])


def _bullets(items: list[str], bullet: str = "•") -> ListFlowable:
    return ListFlowable([ListItem(_p(i), leftIndent=12, value=bullet) for i in items],
                        bulletType="bullet", start=bullet, leftIndent=12, bulletColor=PRIMARY)


def _kpis(items: list[tuple[str, str]]) -> Table:
    cells = [Table([[Paragraph(_safe(label.upper()), STYLES["kpi_label"])],
                    [Paragraph(_safe(value), STYLES["kpi_value"])]],
                   style=[("LEFTPADDING", (0, 0), (-1, -1), 0)])
             for label, value in items]
    row = Table([cells], colWidths=[42 * mm] * len(cells))
    row.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
    ]))
    return row


def _bar_table(rows: list[tuple[str, int, str]], total: int, colour_for=lambda _: PRIMARY) -> Table:
    """A small table with a proportional bar for each row: (label, count, value text)."""
    data, styles = [], []
    for i, (label, count, value) in enumerate(rows):
        width = 70 * count / total if total else 0
        bar = Table([[""]], colWidths=[max(width, 0.1) * mm], rowHeights=[3.2 * mm])
        if count:
            bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colour_for(label))]))
        data.append([_p(label.title() if label.isupper() else label), bar, _p(value, "small")])
        styles.append(("LINEBELOW", (0, i), (-1, i), 0.3, BORDER))
    table = Table(data, colWidths=[38 * mm, 75 * mm, 30 * mm], hAlign="LEFT")
    table.setStyle(TableStyle(styles + [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return table


def build_insights_pdf(session: dict, insights: Insights, model: str, include_transcript: bool = False) -> bytes:
    metrics = session_metrics(session)
    stats = metrics["fillers"]
    buffer = io.BytesIO()
    generated = datetime.now().strftime("%d %b %Y, %H:%M")

    def decorate(canvas, doc):
        width, height = A4
        canvas.saveState()
        canvas.setFillColor(PRIMARY)
        canvas.rect(0, height - 12 * mm, width, 12 * mm, stroke=0, fill=1)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(18 * mm, height - 7.8 * mm, "VoxAuraAI")
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(width - 18 * mm, height - 7.8 * mm, "Recording insights report")
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(18 * mm, 10 * mm, f"Generated {generated} with {model}. AI-generated content - "
                                            "verify important details against the transcript.")
        canvas.drawRightString(width - 18 * mm, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=22 * mm, bottomMargin=18 * mm,
                            title=f"Insights - {insights.title or session['title']}", author="VoxAuraAI")
    source = {"upload": "Uploaded audio", "recording": "Live recording", "youtube": "YouTube",
              "text": "Text transcript"}.get(session.get("source_type"), "Recording")
    story = [
        _p(insights.title or session["title"], "title"),
        Spacer(1, 2),
        _p(f"{source}  ·  {session.get('source_name') or ''}  ·  {session.get('created_at') or generated}", "meta"),
        Spacer(1, 10),
        _kpis([("Duration", metrics["duration"]), ("Words", str(metrics["words"])),
               ("Speaking rate", f"{metrics['wpm']} wpm" if metrics["wpm"] else "-"),
               ("Fillers / 100 words", str(stats.per_100_words))]),
        Paragraph("Summary", STYLES["h2"]), _p(insights.summary or "-"),
    ]
    if insights.tone:
        story += [Paragraph("Tone and speaking style", STYLES["h2"]), _p(insights.tone)]
    story += [Paragraph("Key points", STYLES["h2"]),
              _bullets(insights.key_points) if insights.key_points else _p("None.")]
    story += [Paragraph("Action items", STYLES["h2"]),
              _bullets(insights.action_items, "□") if insights.action_items else _p("No action items mentioned.")]

    speech = [Paragraph("Speech analysis", STYLES["h2"])]
    if stats.counts:
        top = list(stats.counts.items())[:8]
        speech += [_p(f"{stats.total_fillers} filler words in {stats.total_words} words "
                      f"({stats.per_100_words} per 100 words).", "small"), Spacer(1, 3),
                   _bar_table([(k, v, str(v)) for k, v in top], max(v for _, v in top))]
    else:
        speech.append(_p("No filler words detected.", "small"))
    sentiment = metrics["sentiment"]
    if sentiment:
        total = sum(sentiment.values())
        rows = [(k, sentiment.get(k, 0), f"{round(100 * sentiment.get(k, 0) / total)}%")
                for k in ("POSITIVE", "NEUTRAL", "NEGATIVE")]
        speech += [Spacer(1, 6), _p("Sentence sentiment (AssemblyAI model)", "small"), Spacer(1, 3),
                   _bar_table(rows, total, lambda label: colors.HexColor(SENTIMENT_COLOURS[label]))]
    story.append(KeepTogether(speech))

    if insights.suggested_questions:
        story += [Paragraph("Questions to explore", STYLES["h2"]), _bullets(insights.suggested_questions)]
    if include_transcript:
        story += [PageBreak(), Paragraph("Transcript (cleaned)", STYLES["h2"])]
        story += [_p(s["text"]) for s in (session.get("segments") or [{"text": session["text"]}])]

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buffer.getvalue()
