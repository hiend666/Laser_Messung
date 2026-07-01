"""
md_to_pdf.py – Wandelt docs/Anleitung.md in docs/Anleitung.pdf um.

Unterstützt (Teilmenge von Markdown):
  - # ## ### Überschriften
  - Absätze (mit Inline **bold**, *italic*, `code`, „..."-Anführungszeichen)
  - Aufzählungen (- )
  - Markdown-Tabellen (| ... |)
  - Bilder ![Alt](relativer_pfad.png) als zentriertes Bild + Caption
  - Zitat-Zeilen (> )
  - horizontale Trennlinien (---)

Abhängigkeiten: reportlab (im venv vorhanden), re, pathlib.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------------------------
# Inline-Markdown → reportlab-Markup (<b>, <i>, <font>)
# ---------------------------------------------------------------------------
def _escape(text: str) -> str:
    """Escapet & < > für reportlab."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(text: str) -> str:
    """Wandelt Inline-Markdown in reportlab-Paragraph-Markup um."""
    s = _escape(text)
    # Reihenfolge wichtig: zuerst Backtick-Code, dann Bold, dann Italic
    s = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", s)
    return s


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def build_styles() -> dict:
    base = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, spaceBefore=4, spaceAfter=12, textColor=colors.HexColor("#1a3a6c"),
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=15, leading=18, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#1a3a6c"),
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"], fontName="Helvetica-Bold",
            fontSize=12, leading=15, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#2a4a7c"),
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10, leading=14, spaceAfter=6, alignment=TA_LEFT,
        ),
        "list": ParagraphStyle(
            "list", parent=base["BodyText"], fontName="Helvetica",
            fontSize=10, leading=14, leftIndent=14, bulletIndent=2, spaceAfter=3,
        ),
        "quote": ParagraphStyle(
            "quote", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=9.5, leading=13, leftIndent=12, textColor=colors.HexColor("#555555"),
            borderColor=colors.HexColor("#cccccc"), borderWidth=0, spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "caption", parent=base["BodyText"], fontName="Helvetica-Oblique",
            fontSize=9, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#666666"),
            spaceBefore=4, spaceAfter=10,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["BodyText"], fontName="Helvetica",
            fontSize=9, leading=11, spaceAfter=0,
        ),
        "cell_h": ParagraphStyle(
            "cell_h", parent=base["BodyText"], fontName="Helvetica-Bold",
            fontSize=9, leading=11, spaceAfter=0, textColor=colors.white,
        ),
    }
    return styles


# ---------------------------------------------------------------------------
# Markdown-Parser → list of flowables
# ---------------------------------------------------------------------------
def parse_md(md_text: str, base_dir: Path, styles: dict) -> list:
    flow = []
    lines = md_text.splitlines()
    i = 0
    n = len(lines)

    # Maximalbreite für Bilder (Seitenbreite − Margins)
    max_img_w = A4[0] - 3.4 * cm

    def add_paragraph_block(block_lines: list):
        text = " ".join(l.strip() for l in block_lines if l.strip())
        if text:
            flow.append(Paragraph(inline(text), styles["body"]))

    while i < n:
        line = lines[i]
        raw = line.rstrip()

        # Leerzeile -> Paragraphen-Ende (nicht als leerer Spacer)
        if not raw.strip():
            i += 1
            continue

        # Horizontale Trennlinie
        if re.match(r"^-{3,}$", raw):
            flow.append(Spacer(1, 6))
            i += 1
            continue

        # Überschriften
        m = re.match(r"^(#{1,3})\s+(.*)$", raw)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            style_key = f"h{level}"
            flow.append(Paragraph(inline(text), styles[style_key]))
            i += 1
            continue

        # Bild  ![Alt](path)
        m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", raw)
        if m:
            alt, rel = m.group(1), m.group(2)
            img_path = (base_dir / rel).resolve()
            if not img_path.exists():
                flow.append(Paragraph(f"[Bild fehlt: {rel}]", styles["body"]))
                i += 1
                continue
            # Bild proportional skaliert auf max. Seitenbreite; max. Höhe 18 cm
            from PIL import Image as PILImage
            try:
                with PILImage.open(img_path) as im:
                    iw, ih = im.size
                w = max_img_w
                h = w * ih / iw
                max_h = 18 * cm
                if h > max_h:
                    h = max_h
                    w = h * iw / ih
            except Exception:
                w, h = max_img_w, 10 * cm
            try:
                flow.append(Image(str(img_path), width=w, height=h))
                if alt:
                    flow.append(Paragraph(inline(alt), styles["caption"]))
                else:
                    flow.append(Spacer(1, 6))
            except Exception as e:
                flow.append(Paragraph(f"[Bild-Fehler: {e}]", styles["body"]))
            i += 1
            continue

        # Tabelle (Beginn mit |)
        if raw.startswith("|"):
            tbl_lines = []
            while i < n and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            flow.append(_build_table(tbl_lines, styles))
            continue

        # Zitat
        if raw.startswith(">"):
            quote_lines = []
            while i < n and lines[i].lstrip().startswith(">"):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            quote_text = " ".join(l.strip() for l in quote_lines if l.strip())
            if quote_text:
                flow.append(Paragraph(inline(quote_text), styles["quote"]))
            continue

        # Aufzählung
        if re.match(r"^\s*[-*]\s+", raw):
            list_lines = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                item = re.sub(r"^\s*[-*]\s+", "", lines[i]).strip()
                # Ansammlung mehrzeiliger Bullet-Items
                if item:
                    list_lines.append(item)
                i += 1
            for item in list_lines:
                # Bullet mit Bulletchar + Inline
                flow.append(Paragraph(inline(item), styles["list"], bulletText="•"))
            continue

        # Normaler Absatz (sammel bis Leerzeile/Block-Wechsel)
        block = []
        while i < n and lines[i].strip() and not _is_block_start(lines[i]):
            block.append(lines[i])
            i += 1
        add_paragraph_block(block)

    return flow


def _is_block_start(line: str) -> bool:
    """Erkennt, ob eine Zeile einen neuen Block beginnt (Tabelle, Überschrift, Bild, Liste, Trennlinie, Zitat)."""
    s = line.rstrip()
    return (
        s.startswith("|")
        or re.match(r"^#{1,3}\s", s)
        or re.match(r"^!\[", s)
        or re.match(r"^\s*[-*]\s+", s)
        or re.match(r"^-{3,}$", s)
        or s.startswith(">")
    )


def _build_table(tbl_lines: list, styles: dict) -> Table:
    """Baut eine reportlab-Tabelle aus Markdown-Tabellenzeilen."""
    rows = []
    for idx, ln in enumerate(tbl_lines):
        # Trennzeile "|---|---|" überspringen
        if re.match(r"^\|[\s:|-]+\|\s*$", ln):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return Spacer(1, 0)

    # Spaltenanzahl anpassen (auffüllen)
    n_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < n_cols:
            r.append("")

    # Header = erste Zeile, Rest Daten
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []

    # reportlab will Paragraphs für Umbrüche
    header_para = [Paragraph(inline(c), styles["cell_h"]) for c in header]
    body_para = [[Paragraph(inline(c), styles["cell"]) for c in r] for r in body]

    # Relative Spaltenbreiten: erste Spalte etwas schmaler, falls oft kurz
    col_widths = None
    avail = A4[0] - 3.4 * cm
    if n_cols == 2:
        col_widths = [avail * 0.32, avail * 0.68]
    else:
        col_widths = [avail / n_cols] * n_cols

    data = [header_para] + body_para
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a6c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbbbbb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f8")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Header/Footer (Seitennummer)
# ---------------------------------------------------------------------------
def _header_footer(canvas, doc):
    canvas.saveState()
    # Footer: Seitenzahl mittig
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"Seite {doc.page}")
    # Header: Titel
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.drawString(1.7 * cm, A4[1] - 1.0 * cm, "VERMESSdaten Auswertung – Anleitung")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base_dir = Path(__file__).resolve().parent.parent / "docs"
    md_path = base_dir / "Anleitung.md"
    pdf_path = base_dir / "Anleitung.pdf"

    if not md_path.exists():
        print(f"FEHLER: {md_path} nicht gefunden", file=sys.stderr)
        sys.exit(1)

    md_text = md_path.read_text(encoding="utf-8")
    styles = build_styles()
    flow = parse_md(md_text, base_dir, styles)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.7 * cm, bottomMargin=1.7 * cm,
        title="VERMESSdaten Auswertung – Anleitung",
        author="Laservibrometer-Auswertung",
    )
    doc.build(flow, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print(f"PDF erstellt: {pdf_path}")


if __name__ == "__main__":
    main()
