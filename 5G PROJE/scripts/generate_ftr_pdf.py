"""Render the canonical Final Design Report with the official page geometry."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "SİNAPTİC5G_FINAL_TASARIM_RAPORU.md"
OUTPUT = ROOT / "output" / "pdf" / "SINAPTIC5G_Final_Tasarim_Raporu.pdf"


def register_fonts() -> None:
    font_dir = Path("C:/Windows/Fonts")
    pdfmetrics.registerFont(TTFont("Arial", str(font_dir / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Bold", str(font_dir / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Black", str(font_dir / "ariblk.ttf")))
    pdfmetrics.registerFont(TTFont("Arial-Italic", str(font_dir / "ariali.ttf")))


def inline_markup(text: str) -> str:
    text = html.escape(text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Arial-Bold'>\1</font>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def architecture_table(width: float) -> Table:
    labels = ["Video", ">", "ONNX\nmodeller", ">", "Takip +\nagregasyon", ">", "Şema\ndoğrulama", ">", "results.json"]
    widths = [width * 0.14, width * 0.035, width * 0.18, width * 0.035, width * 0.20, width * 0.035, width * 0.18, width * 0.035, width * 0.17]
    row = [Paragraph(item.replace("\n", "<br/>"), STYLES["diagram"]) for item in labels]
    table = Table([row], colWidths=widths, rowHeights=1.2 * cm)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#DCEAF7")),
        ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#E8F1F8")),
        ("BACKGROUND", (4, 0), (4, 0), colors.HexColor("#E8F1F8")),
        ("BACKGROUND", (6, 0), (6, 0), colors.HexColor("#E8F1F8")),
        ("BACKGROUND", (8, 0), (8, 0), colors.HexColor("#DCEAF7")),
        ("BOX", (0, 0), (0, 0), 0.7, colors.HexColor("#1D4E89")),
        ("BOX", (2, 0), (2, 0), 0.7, colors.HexColor("#1D4E89")),
        ("BOX", (4, 0), (4, 0), 0.7, colors.HexColor("#1D4E89")),
        ("BOX", (6, 0), (6, 0), 0.7, colors.HexColor("#1D4E89")),
        ("BOX", (8, 0), (8, 0), 0.7, colors.HexColor("#1D4E89")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def parse_table(lines: list[str], start: int, available_width: float) -> tuple[Table, int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            rows.append(cells)
        index += 1
    columns = max(len(row) for row in rows)
    normalized = [row + [""] * (columns - len(row)) for row in rows]
    data = [[Paragraph(inline_markup(cell), STYLES["table_header"] if r == 0 else STYLES["table_body"]) for cell in row] for r, row in enumerate(normalized)]
    weights = [1.0] * columns
    if columns >= 2:
        weights[0] = 1.5
    if columns == 3:
        weights = [1.35, 0.8, 1.45]
    if columns == 6:
        weights = [1.7, 0.7, 0.65, 0.65, 0.7, 0.75]
    total = sum(weights)
    widths = [available_width * weight / total for weight in weights]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4E89")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#8AA4BE")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F7FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return table, index


def build_story(text: str, available_width: float) -> list:
    lines = text.splitlines()
    story: list = []
    index = 0
    page_segment = 0
    in_code = False
    code_language = ""
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        joined = " ".join(part.strip() for part in paragraph).strip()
        paragraph.clear()
        if not joined:
            return
        style = STYLES["cover_body"] if page_segment == 0 else STYLES["body"]
        story.append(Paragraph(inline_markup(joined), style))
        story.append(Spacer(1, 4))

    while index < len(lines):
        raw = lines[index]
        hard_break = raw.endswith("  ")
        stripped = raw.strip()
        if stripped.startswith("```"):
            flush_paragraph()
            if not in_code:
                in_code = True
                code_language = stripped[3:].strip()
            else:
                if code_language == "mermaid":
                    story.append(architecture_table(available_width))
                    story.append(Spacer(1, 6))
                in_code = False
                code_language = ""
            index += 1
            continue
        if in_code:
            index += 1
            continue
        if stripped == "---":
            flush_paragraph()
            if page_segment < 2:
                story.append(PageBreak())
                page_segment += 1
            index += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table, index = parse_table(lines, index, available_width)
            story.append(table)
            story.append(Spacer(1, 5))
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = inline_markup(heading.group(2))
            if page_segment == 0:
                style = STYLES["cover_title"] if level == 1 else STYLES["cover_subtitle"]
            elif page_segment == 1:
                style = STYLES["toc_title"]
            else:
                style = STYLES[f"h{level}"]
            story.append(Paragraph(title, style))
            story.append(Spacer(1, 5 if level < 3 else 2))
            index += 1
            continue
        if re.match(r"^\d+\.\s", stripped) or stripped.startswith("- "):
            flush_paragraph()
            bullet_text = re.sub(r"^(\d+\.|-)\s+", "", stripped)
            story.append(Paragraph(inline_markup(bullet_text), STYLES["bullet"], bulletText="•"))
            index += 1
            continue
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if hard_break:
            paragraph.append(stripped)
            flush_paragraph()
        else:
            paragraph.append(stripped)
        index += 1
    flush_paragraph()
    return story


def decorate_page(canvas, document) -> None:
    page = canvas.getPageNumber()
    canvas.saveState()
    if page >= 3:
        canvas.setStrokeColor(colors.HexColor("#9DB2C7"))
        canvas.setLineWidth(0.45)
        canvas.line(2.5 * cm, A4[1] - 2.1 * cm, A4[0] - 2.5 * cm, A4[1] - 2.1 * cm)
        canvas.setFont("Arial", 8)
        canvas.setFillColor(colors.HexColor("#4A6075"))
        canvas.drawString(2.5 * cm, A4[1] - 1.85 * cm, "SİNAPTİC5G - Final Tasarım Raporu")
        canvas.drawRightString(A4[0] - 2.5 * cm, A4[1] - 1.85 * cm, "SinapticLink5G")
    if page >= 2:
        canvas.setFont("Arial", 9)
        canvas.setFillColor(colors.HexColor("#4A6075"))
        canvas.drawCentredString(A4[0] / 2, 1.45 * cm, str(page))
    canvas.restoreState()


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle("cover_title", parent=base["Title"], fontName="Arial-Black", fontSize=18, leading=22, alignment=TA_CENTER, textColor=colors.HexColor("#153B65"), spaceBefore=3.5 * cm),
        "cover_subtitle": ParagraphStyle("cover_subtitle", parent=base["Heading2"], fontName="Arial-Black", fontSize=14, leading=18, alignment=TA_CENTER, textColor=colors.HexColor("#1D4E89"), spaceBefore=0.4 * cm),
        "cover_body": ParagraphStyle("cover_body", parent=base["BodyText"], fontName="Arial", fontSize=12, leading=16, alignment=TA_CENTER, spaceBefore=0.35 * cm),
        "toc_title": ParagraphStyle("toc_title", parent=base["Heading1"], fontName="Arial-Black", fontSize=14, leading=17, alignment=TA_LEFT, textColor=colors.HexColor("#153B65"), spaceBefore=0.2 * cm),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Arial-Black", fontSize=14, leading=17, textColor=colors.HexColor("#153B65"), spaceBefore=5, spaceAfter=2, keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Arial-Black", fontSize=14, leading=17, textColor=colors.HexColor("#153B65"), spaceBefore=5, spaceAfter=2, keepWithNext=True),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName="Arial-Bold", fontSize=12, leading=14, textColor=colors.HexColor("#1D4E89"), spaceBefore=4, spaceAfter=1, keepWithNext=True),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontName="Arial", fontSize=12, leading=13.8, alignment=TA_JUSTIFY, textColor=colors.HexColor("#17212B"), splitLongWords=True),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontName="Arial", fontSize=11, leading=12.8, leftIndent=13, firstLineIndent=-7, alignment=TA_JUSTIFY, bulletFontName="Arial"),
        "table_header": ParagraphStyle("table_header", parent=base["BodyText"], fontName="Arial-Bold", fontSize=7.2, leading=8.3, textColor=colors.white, alignment=TA_CENTER),
        "table_body": ParagraphStyle("table_body", parent=base["BodyText"], fontName="Arial", fontSize=7.1, leading=8.2, alignment=TA_LEFT),
        "diagram": ParagraphStyle("diagram", parent=base["BodyText"], fontName="Arial-Bold", fontSize=7.6, leading=9, alignment=TA_CENTER, textColor=colors.HexColor("#153B65")),
    }


register_fonts()
STYLES = make_styles()


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=2.5 * cm,
        leftMargin=2.5 * cm,
        topMargin=2.8 * cm,
        bottomMargin=2.5 * cm,
        title="SİNAPTİC5G Final Tasarım Raporu",
        author="SinapticLink5G",
        subject="TEKNOFEST 2026 5G ve Yapay Zekâ ile Akıllı Yol Güvenliği",
    )
    story = build_story(SOURCE.read_text(encoding="utf-8"), document.width)
    document.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
