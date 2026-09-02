"""Generate the deterministic Project Aurora demonstration PDF and ground truth."""

from __future__ import annotations

import hashlib
import json
import random
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = PROJECT_ROOT / "demo" / "source" / "project_aurora_mission_readiness_report.pdf"
GROUND_TRUTH_PATH = PROJECT_ROOT / "demo" / "ground_truth.json"

PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 42
INK = colors.HexColor("#172033")
NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#2E75B6")
PALE_BLUE = colors.HexColor("#DCEAF7")
PALE_GOLD = colors.HexColor("#FFF2CC")
PALE_RED = colors.HexColor("#FCE4D6")
GRID = colors.HexColor("#8194A8")
MUTED = colors.HexColor("#5B6573")


GROUND_TRUTH: list[dict[str, Any]] = [
    {
        "id": "Q01",
        "question": "What verification method is specified for REQ-204?",
        "answerable": True,
        "accepted_answers": ["test", "verified by test", "end-to-end test"],
        "expected_pages": [4],
        "expected_kind": "table",
        "expected_terms": ["REQ-204", "Communications", "Test"],
    },
    {
        "id": "Q02",
        "question": "Which subsystem owns REQ-205 and how is it verified?",
        "answerable": True,
        "accepted_answers": ["navigation; analysis", "navigation by analysis", "Navigation and Analysis"],
        "expected_pages": [4],
        "expected_kind": "table",
        "expected_terms": ["REQ-205", "Navigation", "Analysis"],
    },
    {
        "id": "Q03",
        "question": "What thermal margin threshold is required by REQ-207?",
        "answerable": True,
        "accepted_answers": ["18 percent", "18%", "a thermal margin of 18 percent"],
        "expected_pages": [5],
        "expected_kind": "table",
        "expected_terms": ["REQ-207", "thermal", "18 percent"],
    },
    {
        "id": "Q04",
        "question": "What is the readiness status of REQ-209?",
        "answerable": True,
        "accepted_answers": ["conditional", "status is conditional"],
        "expected_pages": [5],
        "expected_kind": "table",
        "expected_terms": ["REQ-209", "Conditional"],
    },
    {
        "id": "Q05",
        "question": "How much scheduled maintenance is permitted each quarter?",
        "answerable": True,
        "accepted_answers": ["four hours per quarter", "4 hours per quarter", "no more than four hours"],
        "expected_pages": [3],
        "expected_kind": "list",
        "expected_terms": ["scheduled maintenance", "four hours per quarter"],
    },
    {
        "id": "Q06",
        "question": "What readiness percentage was achieved in Q4?",
        "answerable": True,
        "accepted_answers": ["92 percent", "92%"],
        "expected_pages": [6],
        "expected_kind": "picture",
        "expected_terms": ["Q4", "92 percent"],
    },
    {
        "id": "Q07",
        "question": "Where does the Sensor Gateway send telemetry?",
        "answerable": True,
        "accepted_answers": ["Validation Service", "the Validation Service"],
        "expected_pages": [7],
        "expected_kind": "picture",
        "expected_terms": ["Sensor Gateway", "telemetry", "Validation Service"],
    },
    {
        "id": "Q08",
        "question": "What recovery window is stated in the scanned appendix?",
        "answerable": True,
        "accepted_answers": ["15 minutes", "a 15-minute recovery window"],
        "expected_pages": [8],
        "expected_kind": "text",
        "expected_terms": ["recovery window", "15 minutes"],
    },
    {
        "id": "Q09",
        "question": "What is the document identifier?",
        "answerable": True,
        "accepted_answers": ["AUR-MRA-001"],
        "expected_pages": [1],
        "expected_kind": "text",
        "expected_terms": ["Document ID", "AUR-MRA-001"],
    },
    {
        "id": "Q10",
        "question": "What revision is this assessment?",
        "answerable": True,
        "accepted_answers": ["C", "revision C"],
        "expected_pages": [1],
        "expected_kind": "text",
        "expected_terms": ["Revision", "C"],
    },
    {
        "id": "Q11",
        "question": "When was the assessment released?",
        "answerable": True,
        "accepted_answers": ["August 28, 2026", "2026-08-28"],
        "expected_pages": [1],
        "expected_kind": "text",
        "expected_terms": ["Release Date", "August 28, 2026"],
    },
    {
        "id": "Q12",
        "question": "Which requirement identifier appears in the thermal-margin code sample?",
        "answerable": True,
        "accepted_answers": ["REQ-207"],
        "expected_pages": [9],
        "expected_kind": "code",
        "expected_terms": ["thermal_margin", "REQ-207"],
    },
    {
        "id": "Q13",
        "question": "What was the contract award date?",
        "answerable": False,
        "accepted_answers": ["not found", "insufficient evidence", "not stated"],
        "expected_pages": [],
        "expected_kind": "unsupported",
        "expected_terms": [],
    },
    {
        "id": "Q14",
        "question": "Who is the program's chief executive?",
        "answerable": False,
        "accepted_answers": ["not found", "insufficient evidence", "not stated"],
        "expected_pages": [],
        "expected_kind": "unsupported",
        "expected_terms": [],
    },
]


def _set_document_metadata(pdf: Canvas) -> None:
    pdf.setTitle("Project Aurora Mission Readiness Assessment")
    pdf.setAuthor("Parse Before You Prompt Demo")
    pdf.setSubject("Deterministic synthetic document for parsing and retrieval evaluation")
    pdf.setCreator("ReportLab, Matplotlib, and Pillow")


def _header_footer(pdf: Canvas, page_number: int, section: str) -> None:
    pdf.setStrokeColor(GRID)
    pdf.setLineWidth(0.5)
    pdf.line(MARGIN, PAGE_HEIGHT - 34, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 34)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 27, "AUR-MRA-001  |  Revision C")
    pdf.drawRightString(PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 27, section)
    pdf.line(MARGIN, 31, PAGE_WIDTH - MARGIN, 31)
    pdf.drawString(MARGIN, 20, "Project Aurora Mission Readiness Assessment")
    pdf.drawRightString(PAGE_WIDTH - MARGIN, 20, f"Page {page_number} of 10")


def _draw_wrapped(
    pdf: Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font: str = "Helvetica",
    size: float = 10,
    leading: float = 14,
    color: colors.Color = INK,
) -> float:
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for paragraph in text.split("\n"):
        if not paragraph:
            y -= leading
            continue
        for line in simpleSplit(paragraph, font, size, width):
            pdf.drawString(x, y, line)
            y -= leading
    return y


def _draw_bullet(
    pdf: Canvas, text: str, x: float, y: float, width: float, *, level: int = 0
) -> float:
    indent = level * 16
    marker = "-" if level else "•"
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(BLUE)
    pdf.drawString(x + indent, y, marker)
    return _draw_wrapped(pdf, text, x + indent + 14, y, width - indent - 14, size=10, leading=14)


def _page_cover(pdf: Canvas) -> None:
    pdf.setFillColor(NAVY)
    pdf.rect(0, PAGE_HEIGHT - 190, PAGE_WIDTH, 190, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 28)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 88, "PROJECT AURORA")
    pdf.setFont("Helvetica", 19)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 123, "Mission Readiness Assessment")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 154, "Synthetic demonstration report — controlled evaluation copy")

    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 255, "DOCUMENT CONTROL")
    details = [
        ("Document ID", "AUR-MRA-001"),
        ("Revision", "C"),
        ("Release Date", "August 28, 2026"),
        ("Classification", "Synthetic / Public Demonstration"),
    ]
    y = PAGE_HEIGHT - 290
    for label, value in details:
        pdf.setFillColor(PALE_BLUE)
        pdf.rect(MARGIN, y - 7, 145, 28, fill=1, stroke=0)
        pdf.setFillColor(INK)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(MARGIN + 8, y + 3, label)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(MARGIN + 165, y + 3, value)
        y -= 38

    pdf.setFillColor(PALE_GOLD)
    pdf.roundRect(MARGIN, 180, PAGE_WIDTH - 2 * MARGIN, 94, 8, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN + 14, 248, "Purpose")
    _draw_wrapped(
        pdf,
        "This original, copyright-safe report was created to compare plain-text extraction with structure-aware document parsing. All names, identifiers, systems, and measurements are fictional.",
        MARGIN + 14,
        228,
        PAGE_WIDTH - 2 * MARGIN - 28,
        size=10,
        leading=14,
    )
    _header_footer(pdf, 1, "Cover and document control")
    pdf.showPage()


def _page_executive_summary(pdf: Canvas) -> None:
    _header_footer(pdf, 2, "Executive summary")
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 68, "Executive Summary")

    gutter = 22
    column_width = (PAGE_WIDTH - 2 * MARGIN - gutter) / 2
    left_x = MARGIN
    right_x = MARGIN + column_width + gutter
    left_y = PAGE_HEIGHT - 102
    right_y = PAGE_HEIGHT - 102

    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(BLUE)
    pdf.drawString(left_x, left_y, "Assessment posture")
    left_y -= 22
    left_y = _draw_wrapped(
        pdf,
        "Project Aurora has entered integrated mission-readiness review. The assessment combines requirement verification, operational constraints, telemetry validation, and recovery planning into one controlled baseline.",
        left_x,
        left_y,
        column_width,
        size=10,
        leading=14,
    )
    left_y -= 12
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(BLUE)
    pdf.drawString(left_x, left_y, "Decision context")
    left_y -= 22
    left_y = _draw_wrapped(
        pdf,
        "Readiness improved through four quarterly reviews. Open work is concentrated in one conditional navigation item and does not change the verified communications evidence recorded in the matrix.",
        left_x,
        left_y,
        column_width,
        size=10,
        leading=14,
    )
    left_y -= 14
    left_y = _draw_bullet(pdf, "Core telemetry interfaces are available for integrated validation.", left_x, left_y, column_width)
    left_y -= 5
    left_y = _draw_bullet(pdf, "Recovery procedures have been rehearsed against the continuity checklist.", left_x, left_y, column_width)
    left_y -= 5
    _draw_bullet(pdf, "Requirement evidence remains traceable to the verification matrix.", left_x, left_y, column_width)

    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(BLUE)
    pdf.drawString(right_x, right_y, "Scope and exclusions")
    right_y -= 22
    right_y = _draw_wrapped(
        pdf,
        "This revision covers mission-readiness evidence through Q4 2026. It does not contain commercial award records, executive biographies, personnel compensation, or supplier pricing.",
        right_x,
        right_y,
        column_width,
        size=10,
        leading=14,
    )
    right_y -= 14
    pdf.setFillColor(PALE_GOLD)
    callout_height = 126
    pdf.roundRect(right_x, right_y - callout_height + 12, column_width, callout_height, 7, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(right_x + 12, right_y - 10, "READINESS CALLOUT")
    _draw_wrapped(
        pdf,
        "The evidence baseline supports continued integrated validation. Final disposition depends on closing the conditional navigation demonstration identified as REQ-209.",
        right_x + 12,
        right_y - 34,
        column_width - 24,
        size=10,
        leading=14,
    )
    right_y -= callout_height + 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.setFillColor(BLUE)
    pdf.drawString(right_x, right_y, "Review principle")
    right_y -= 22
    _draw_wrapped(
        pdf,
        "A readiness claim is accepted only when its requirement, method, owner, status, and evidence can be traced together.",
        right_x,
        right_y,
        column_width,
        size=10,
        leading=14,
    )
    pdf.showPage()


def _page_requirements(pdf: Canvas) -> None:
    _header_footer(pdf, 3, "Readiness criteria")
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 68, "1. Readiness Criteria")

    y = PAGE_HEIGHT - 102
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(MARGIN, y, "1.1 Operational Availability")
    y -= 25
    y = _draw_wrapped(
        pdf,
        "Mission services shall sustain planned operations while preserving bounded maintenance windows and recoverable state transitions.",
        MARGIN,
        y,
        PAGE_WIDTH - 2 * MARGIN,
        size=10,
        leading=14,
    )
    y -= 10
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(MARGIN + 14, y, "1.1.1 Maintenance Constraints")
    y -= 23
    y = _draw_bullet(
        pdf,
        "Scheduled maintenance must not exceed four hours per quarter.",
        MARGIN + 14,
        y,
        PAGE_WIDTH - 2 * MARGIN - 14,
    )
    y -= 5
    y = _draw_bullet(
        pdf,
        "Emergency maintenance requires an incident record and readiness-board review.",
        MARGIN + 14,
        y,
        PAGE_WIDTH - 2 * MARGIN - 14,
    )
    y -= 12
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(MARGIN + 14, y, "1.1.2 Recovery Objectives")
    y -= 23
    y = _draw_bullet(pdf, "Preserve the last validated telemetry checkpoint.", MARGIN + 14, y, PAGE_WIDTH - 2 * MARGIN - 14)
    y -= 5
    y = _draw_bullet(pdf, "Record recovery timing from initial alert to service validation.", MARGIN + 14, y, PAGE_WIDTH - 2 * MARGIN - 14)

    y -= 18
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(MARGIN, y, "1.2 Verification Governance")
    y -= 25
    y = _draw_wrapped(
        pdf,
        "Each requirement is assigned one primary verification method and a named evidence owner. Conditional items remain visible until their closure evidence is accepted.",
        MARGIN,
        y,
        PAGE_WIDTH - 2 * MARGIN,
        size=10,
        leading=14,
    )
    y -= 10
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(MARGIN + 14, y, "1.2.1 Evidence Acceptance")
    y -= 23
    y = _draw_bullet(pdf, "Test evidence includes repeatable inputs and observed outputs.", MARGIN + 14, y, PAGE_WIDTH - 2 * MARGIN - 14)
    y -= 5
    y = _draw_bullet(pdf, "Analysis evidence records assumptions and calculation bounds.", MARGIN + 14, y, PAGE_WIDTH - 2 * MARGIN - 14)
    y -= 5
    _draw_bullet(pdf, "Demonstration evidence records operator steps and witnessed results.", MARGIN + 14, y, PAGE_WIDTH - 2 * MARGIN - 14)
    pdf.showPage()


def _cell(text: str, *, bold: bool = False, align: int = TA_LEFT) -> Paragraph:
    style = ParagraphStyle(
        "table-cell",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=7.6,
        leading=9.2,
        alignment=align,
        textColor=INK,
    )
    return Paragraph(text, style)


def _draw_verification_table(
    pdf: Canvas,
    rows: list[list[str]],
    *,
    page_number: int,
    continuation: bool,
    row_spans: list[tuple[int, int]],
) -> None:
    _header_footer(pdf, page_number, "Verification matrix")
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 20)
    title = "2. Verification Matrix" if not continuation else "2. Verification Matrix — Continued"
    pdf.drawString(MARGIN, PAGE_HEIGHT - 68, title)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        MARGIN,
        PAGE_HEIGHT - 87,
        "Controlled matrix; grouped headers and subsystem row spans are intentional parsing challenges.",
    )

    data: list[list[Any]] = [
        [
            _cell("Requirement", bold=True, align=TA_CENTER),
            _cell("Subsystem", bold=True, align=TA_CENTER),
            _cell("Verification", bold=True, align=TA_CENTER),
            "",
            _cell("Readiness", bold=True, align=TA_CENTER),
            "",
        ],
        [
            _cell("ID", bold=True, align=TA_CENTER),
            _cell("Domain", bold=True, align=TA_CENTER),
            _cell("Method", bold=True, align=TA_CENTER),
            _cell("Owner", bold=True, align=TA_CENTER),
            _cell("Status", bold=True, align=TA_CENTER),
            _cell("Evidence / criterion", bold=True, align=TA_CENTER),
        ],
    ]
    data.extend([[_cell(value, bold=index == 0) if value else "" for index, value in enumerate(row)] for row in rows])

    table = Table(
        data,
        colWidths=[58, 78, 69, 76, 66, 181],
        rowHeights=[26, 28] + [38] * len(rows),
        repeatRows=2,
        hAlign="LEFT",
    )
    style_commands: list[tuple[Any, ...]] = [
        ("SPAN", (0, 0), (0, 1)),
        ("SPAN", (1, 0), (1, 1)),
        ("SPAN", (2, 0), (3, 0)),
        ("SPAN", (4, 0), (5, 0)),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("BACKGROUND", (0, 1), (-1, 1), PALE_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.55, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for start_row, end_row in row_spans:
        style_commands.append(("SPAN", (1, start_row + 2), (1, end_row + 2)))
        style_commands.append(("BACKGROUND", (1, start_row + 2), (1, end_row + 2), colors.HexColor("#EEF4FA")))
    table.setStyle(TableStyle(style_commands))
    _, table_height = table.wrapOn(pdf, PAGE_WIDTH - 2 * MARGIN, PAGE_HEIGHT)
    table.drawOn(pdf, MARGIN, PAGE_HEIGHT - 112 - table_height)

    note_y = PAGE_HEIGHT - 128 - table_height
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 8)
    if page_number == 4:
        pdf.drawString(MARGIN, note_y, "* REQ-204 communications verification uses an end-to-end Test.")
        pdf.setFont("Helvetica", 8)
        pdf.drawString(MARGIN, note_y - 13, "Matrix continues on page 5; header bands repeat to preserve verification context.")
    else:
        pdf.drawString(MARGIN, note_y, "† REQ-207 acceptance requires a thermal margin threshold of 18 percent.")
        pdf.setFont("Helvetica", 8)
        pdf.drawString(MARGIN, note_y - 13, "‡ REQ-209 remains Conditional pending witnessed closure of the navigation demonstration.")
    pdf.showPage()


def _page_verification_matrix_one(pdf: Canvas) -> None:
    rows = [
        ["REQ-201", "Flight Software", "Analysis", "Avionics", "Ready", "Timing budget reconciled"],
        ["REQ-202", "", "Test", "Avionics", "Ready", "Restart sequence passed"],
        ["REQ-203", "Communications", "Inspection", "Comms", "Ready", "Interface record accepted"],
        ["REQ-204", "", "Test", "Comms", "Ready", "End-to-end telemetry test*"],
        ["REQ-205", "Navigation", "Analysis", "Guidance", "Ready", "Covariance analysis accepted"],
        ["REQ-206", "Thermal", "Test", "Thermal", "Ready", "Chamber profile complete"],
    ]
    _draw_verification_table(pdf, rows, page_number=4, continuation=False, row_spans=[(0, 1), (2, 3)])


def _page_verification_matrix_two(pdf: Canvas) -> None:
    rows = [
        ["REQ-207", "Thermal", "Analysis", "Thermal", "Ready", "Margin threshold: 18 percent†"],
        ["REQ-208", "", "Test", "Thermal", "Ready", "Heater failover passed"],
        ["REQ-209", "Navigation", "Demonstration", "Guidance", "Conditional", "Recovery retest pending‡"],
        ["REQ-210", "", "Inspection", "Guidance", "Ready", "Waypoint set reviewed"],
        ["REQ-211", "Operations", "Analysis", "Support", "Ready", "Maintenance ≤ 4 h/quarter"],
        ["REQ-212", "Safety", "Test", "Safety", "Ready", "Abort annunciator verified"],
    ]
    _draw_verification_table(pdf, rows, page_number=5, continuation=True, row_spans=[(0, 1), (2, 3)])


def _chart_image() -> BytesIO:
    figure = Figure(figsize=(7.0, 3.1), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    readiness = [74, 81, 87, 92]
    bars = axis.bar(quarters, readiness, color=["#9DC3E6", "#5B9BD5", "#4472C4", "#17365D"])
    axis.set_ylim(0, 100)
    axis.set_ylabel("Readiness (%)")
    axis.set_title("Quarterly Integrated Readiness")
    axis.grid(axis="y", color="#D9E2F3", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, readiness, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value}%", ha="center", fontsize=9)
    figure.tight_layout()
    stream = BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=150,
        metadata={"Software": "Parse Before You Prompt Demo"},
    )
    stream.seek(0)
    return stream


def _page_readiness_chart(pdf: Canvas) -> None:
    _header_footer(pdf, 6, "Readiness trend")
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 68, "3. Quarterly Readiness Trend")
    _draw_wrapped(
        pdf,
        "Integrated readiness increased after telemetry reconciliation, recovery rehearsal, and verification-evidence review. The Q4 readiness value is 92 percent.",
        MARGIN,
        PAGE_HEIGHT - 94,
        PAGE_WIDTH - 2 * MARGIN,
        size=10,
        leading=14,
    )
    chart = _chart_image()
    pdf.drawImage(ImageReader(chart), MARGIN, 285, width=PAGE_WIDTH - 2 * MARGIN, height=300, preserveAspectRatio=True)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.setFillColor(INK)
    pdf.drawCentredString(PAGE_WIDTH / 2, 266, "Figure 1. Quarterly readiness increased from 74 percent to 92 percent.")
    _draw_wrapped(
        pdf,
        "The trend is a status indicator, not a substitute for requirement-level evidence. REQ-209 remains visible as Conditional despite the aggregate improvement.",
        MARGIN,
        232,
        PAGE_WIDTH - 2 * MARGIN,
        size=10,
        leading=14,
    )
    pdf.showPage()


def _centered_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=6, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.multiline_text(
        ((left + right - width) / 2, (top + bottom - height) / 2),
        text,
        font=font,
        fill="#172033",
        spacing=6,
        align="center",
    )


def _architecture_image() -> BytesIO:
    image = Image.new("RGB", (1200, 650), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=32)
    box_font = ImageFont.load_default(size=27)
    label_font = ImageFont.load_default(size=22)
    draw.text((40, 30), "Project Aurora Telemetry Validation Architecture", font=title_font, fill="#17365D")

    boxes = {
        "sensor": (70, 210, 330, 365),
        "validation": (470, 210, 760, 365),
        "store": (900, 210, 1130, 365),
        "console": (470, 455, 760, 585),
    }
    fills = {"sensor": "#DCEAF7", "validation": "#FFF2CC", "store": "#E2F0D9", "console": "#F2F2F2"}
    labels = {
        "sensor": "Sensor\nGateway",
        "validation": "Validation\nService",
        "store": "Readiness\nStore",
        "console": "Control\nConsole",
    }
    for key, box in boxes.items():
        draw.rounded_rectangle(box, radius=18, fill=fills[key], outline="#17365D", width=4)
        _centered_text(draw, box, labels[key], box_font)

    def arrow(start: tuple[int, int], end: tuple[int, int], label: str, label_xy: tuple[int, int]) -> None:
        draw.line((start, end), fill="#2E75B6", width=7)
        ex, ey = end
        draw.polygon([(ex, ey), (ex - 20, ey - 12), (ex - 20, ey + 12)], fill="#2E75B6")
        draw.text(label_xy, label, font=label_font, fill="#172033")

    arrow((330, 287), (470, 287), "telemetry", (348, 245))
    arrow((760, 287), (900, 287), "validated events", (770, 245))
    draw.line((615, 455, 615, 365), fill="#2E75B6", width=7)
    draw.polygon([(615, 365), (603, 385), (627, 385)], fill="#2E75B6")
    draw.text((640, 405), "control", font=label_font, fill="#172033")

    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    stream.seek(0)
    return stream


def _page_architecture(pdf: Canvas) -> None:
    _header_footer(pdf, 7, "System architecture")
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 68, "4. Telemetry Validation Architecture")
    _draw_wrapped(
        pdf,
        "The Sensor Gateway sends telemetry to the Validation Service. Validated events are persisted in the Readiness Store, while the Control Console supplies operator-directed control inputs.",
        MARGIN,
        PAGE_HEIGHT - 94,
        PAGE_WIDTH - 2 * MARGIN,
        size=10,
        leading=14,
    )
    diagram = _architecture_image()
    pdf.drawImage(ImageReader(diagram), MARGIN, 292, width=PAGE_WIDTH - 2 * MARGIN, height=330, preserveAspectRatio=True)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(PAGE_WIDTH / 2, 271, "Figure 2. Sensor Gateway to Validation Service telemetry flow.")
    _draw_wrapped(
        pdf,
        "Interface boundary: telemetry is evaluated before readiness records are updated. Control messages follow a separate path and do not bypass validation.",
        MARGIN,
        236,
        PAGE_WIDTH - 2 * MARGIN,
        size=10,
        leading=14,
    )
    pdf.showPage()


def _scanned_appendix_image() -> BytesIO:
    rng = random.Random(20260828)
    width, height = 1650, 2200
    image = Image.new("RGB", (width, height), "#F6F2E7")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.load_default(size=54)
    heading_font = ImageFont.load_default(size=43)
    body_font = ImageFont.load_default(size=35)
    stamp_font = ImageFont.load_default(size=31)

    draw.rectangle((100, 90, width - 100, height - 100), outline="#555555", width=5)
    draw.text((150, 150), "APPENDIX C - CONTINUITY CHECKLIST", font=title_font, fill="#202020")
    draw.line((150, 230, width - 150, 230), fill="#555555", width=4)
    draw.text((150, 300), "C.1 RECOVERY VALIDATION", font=heading_font, fill="#202020")
    scan_lines = [
        "1. Confirm the last validated telemetry checkpoint.",
        "2. Isolate the affected mission service.",
        "3. Restore the service from the approved checkpoint.",
        "4. Verify telemetry through the Validation Service.",
        "5. Record elapsed time and operator initials.",
        "",
        "Maximum recovery window: 15 minutes.",
        "",
        "Acceptance requires two consecutive clean telemetry cycles.",
        "Escalate any incomplete recovery to the readiness board.",
    ]
    y = 390
    for line in scan_lines:
        draw.text((175, y), line, font=body_font, fill="#252525")
        y += 92 if line else 55

    draw.rectangle((1010, 1710, 1450, 1900), outline="#8B2F2F", width=7)
    draw.text((1055, 1770), "SCANNED COPY", font=stamp_font, fill="#8B2F2F")
    draw.text((1064, 1825), "AUR-MRA-001-C", font=stamp_font, fill="#8B2F2F")
    draw.text((155, 1980), "Operator initials: __________   Review date: __________", font=body_font, fill="#252525")

    for _ in range(5000):
        x = rng.randrange(105, width - 105)
        y = rng.randrange(105, height - 105)
        shade = rng.choice((170, 185, 200, 215, 225))
        draw.point((x, y), fill=(shade, shade, shade))

    image = image.rotate(0.75, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="#E8E3D6")
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    stream.seek(0)
    return stream


def _page_scanned_appendix(pdf: Canvas) -> None:
    _header_footer(pdf, 8, "Scanned appendix")
    scan = _scanned_appendix_image()
    pdf.drawImage(
        ImageReader(scan),
        47,
        48,
        width=PAGE_WIDTH - 94,
        height=PAGE_HEIGHT - 96,
        preserveAspectRatio=True,
        anchor="c",
    )
    pdf.showPage()


def _formula_image() -> BytesIO:
    figure = Figure(figsize=(7.2, 1.35), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.axis("off")
    axis.text(
        0.5,
        0.62,
        r"$M_{thermal}=\frac{T_{limit}-T_{observed}}{T_{limit}}\times100=18\%$",
        fontsize=24,
        ha="center",
        va="center",
        color="#17365D",
    )
    axis.text(0.5, 0.12, "REQ-207 acceptance expression", fontsize=10, ha="center", color="#5B6573")
    figure.tight_layout(pad=0.2)
    stream = BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=150,
        metadata={"Software": "Parse Before You Prompt Demo"},
    )
    stream.seek(0)
    return stream


def _page_formula_code(pdf: Canvas) -> None:
    _header_footer(pdf, 9, "Formula and code evidence")
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 68, "5. Thermal Margin Control")
    _draw_wrapped(
        pdf,
        "REQ-207 defines the minimum acceptable thermal margin. The calculation and enforcement example below are retained as searchable technical evidence.",
        MARGIN,
        PAGE_HEIGHT - 94,
        PAGE_WIDTH - 2 * MARGIN,
        size=10,
        leading=14,
    )

    formula = _formula_image()
    pdf.setStrokeColor(GRID)
    pdf.setFillColor(colors.white)
    pdf.roundRect(MARGIN, 493, PAGE_WIDTH - 2 * MARGIN, 125, 6, fill=1, stroke=1)
    pdf.drawImage(ImageReader(formula), MARGIN + 8, 503, width=PAGE_WIDTH - 2 * MARGIN - 16, height=102, preserveAspectRatio=True)

    pdf.setFillColor(PALE_RED)
    pdf.roundRect(MARGIN, 404, PAGE_WIDTH - 2 * MARGIN, 66, 6, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#9C0006"))
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN + 12, 447, "WARNING — THERMAL READINESS HOLD")
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica", 9.5)
    pdf.drawString(MARGIN + 12, 426, "Do not authorize launch when the thermal margin is below 18 percent.")

    pdf.setFillColor(colors.HexColor("#20252B"))
    pdf.roundRect(MARGIN, 218, PAGE_WIDTH - 2 * MARGIN, 156, 5, fill=1, stroke=0)
    code_lines = [
        "def enforce_thermal_margin(thermal_margin: float) -> None:",
        "    required_margin = 0.18",
        "    if thermal_margin < required_margin:",
        "        raise ReadinessHold(\"REQ-207\")",
        "    record_verification(\"REQ-207\", status=\"ready\")",
    ]
    pdf.setFillColor(colors.HexColor("#E8EEF2"))
    pdf.setFont("Courier", 9.5)
    y = 344
    for line in code_lines:
        pdf.drawString(MARGIN + 14, y, line)
        y -= 23
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica-Oblique", 8)
    pdf.drawString(MARGIN, 194, "* The code is illustrative control logic; REQ-207 remains the authoritative threshold source.")
    pdf.showPage()


def _page_references(pdf: Canvas) -> None:
    _header_footer(pdf, 10, "References and distractors")
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(MARGIN, PAGE_HEIGHT - 68, "6. References and Administrative Notes")

    y = PAGE_HEIGHT - 105
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(MARGIN, y, "6.1 Synthetic References")
    y -= 24
    references = [
        "[R1] AUR-STD-011, Telemetry Frame Definitions, Revision B, 2026.",
        "[R2] AUR-OPS-022, Continuity and Recovery Handbook, Revision D, 2026.",
        "[R3] AUR-VAL-118, Integrated Validation Evidence Index, Revision A, 2026.",
        "[R4] AUR-THM-207, Thermal Margin Analysis Note, Revision C, 2026.",
    ]
    for reference in references:
        y = _draw_wrapped(pdf, reference, MARGIN + 10, y, PAGE_WIDTH - 2 * MARGIN - 10, size=9.5, leading=14)
        y -= 5

    y -= 10
    pdf.setFillColor(BLUE)
    pdf.setFont("Helvetica-Bold", 13)
    pdf.drawString(MARGIN, y, "6.2 Administrative Archive Notes")
    y -= 25
    y = _draw_wrapped(
        pdf,
        "The North Annex repaint was completed on June 4, 2024. The archive work order used coating batch SL-442 and closed without corrective action.",
        MARGIN,
        y,
        PAGE_WIDTH - 2 * MARGIN,
        size=10,
        leading=14,
    )
    y -= 11
    y = _draw_wrapped(
        pdf,
        "Visitor badge inventory totals 480 units: 320 blue badges, 120 gray badges, and 40 temporary orange badges.",
        MARGIN,
        y,
        PAGE_WIDTH - 2 * MARGIN,
        size=10,
        leading=14,
    )
    y -= 11
    y = _draw_wrapped(
        pdf,
        "Training simulator room carpeting is slate blue. The facilities note applies only to room finishes and has no bearing on mission readiness.",
        MARGIN,
        y,
        PAGE_WIDTH - 2 * MARGIN,
        size=10,
        leading=14,
    )

    y -= 24
    pdf.setFillColor(PALE_GOLD)
    pdf.roundRect(MARGIN, y - 82, PAGE_WIDTH - 2 * MARGIN, 92, 6, fill=1, stroke=0)
    pdf.setFillColor(INK)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(MARGIN + 12, y - 10, "Evaluation note")
    _draw_wrapped(
        pdf,
        "Administrative details are intentional retrieval distractors. Readiness answers should rely on the relevant requirement, chart, diagram, appendix, or technical control evidence.",
        MARGIN + 12,
        y - 31,
        PAGE_WIDTH - 2 * MARGIN - 24,
        size=9.5,
        leading=13,
    )
    pdf.showPage()


def generate_pdf(output_path: Path = PDF_PATH) -> Path:
    """Generate the ten-page deterministic source document."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = Canvas(
        str(output_path),
        pagesize=letter,
        pageCompression=1,
        invariant=1,
        lang="en-US",
    )
    _set_document_metadata(pdf)
    _page_cover(pdf)
    _page_executive_summary(pdf)
    _page_requirements(pdf)
    _page_verification_matrix_one(pdf)
    _page_verification_matrix_two(pdf)
    _page_readiness_chart(pdf)
    _page_architecture(pdf)
    _page_scanned_appendix(pdf)
    _page_formula_code(pdf)
    _page_references(pdf)
    pdf.save()
    return output_path


def write_ground_truth(output_path: Path = GROUND_TRUTH_PATH) -> Path:
    """Write the deterministic evaluation question set."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(GROUND_TRUTH, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    pdf_path = generate_pdf()
    ground_truth_path = write_ground_truth()
    print(f"Generated {pdf_path.relative_to(PROJECT_ROOT)}")
    print(f"PDF SHA-256: {_sha256(pdf_path)}")
    print(f"Wrote {ground_truth_path.relative_to(PROJECT_ROOT)} ({len(GROUND_TRUTH)} questions)")


if __name__ == "__main__":
    main()
