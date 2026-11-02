
from pathlib import Path
import re
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "research_report.md"
OUTPUT = ROOT / "docs" / "research_report.pdf"
EQUATION_DIR = ROOT / "docs" / "equation_images"

# ---------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=23,
    leading=29,
    alignment=TA_CENTER,
    textColor=colors.HexColor("#17365D"),
    spaceAfter=14,
))

styles.add(ParagraphStyle(
    name="Section",
    parent=styles["Heading1"],
    fontSize=16,
    leading=20,
    textColor=colors.HexColor("#17365D"),
    spaceBefore=16,
    spaceAfter=9,
    keepWithNext=True,
))

styles.add(ParagraphStyle(
    name="Subsection",
    parent=styles["Heading2"],
    fontSize=12,
    leading=15,
    textColor=colors.HexColor("#24527A"),
    spaceBefore=11,
    spaceAfter=6,
    keepWithNext=True,
))

styles.add(ParagraphStyle(
    name="BodyReport",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.2,
    leading=13.5,
    spaceAfter=7,
    alignment=4,
))

styles.add(ParagraphStyle(
    name="TableCell",
    parent=styles["BodyText"],
    fontSize=8,
    leading=10,
))

styles.add(ParagraphStyle(
    name="CodeBlock",
    parent=styles["Code"],
    fontName="Courier",
    fontSize=7.5,
    leading=10,
    leftIndent=8,
    rightIndent=8,
    backColor=colors.HexColor("#F2F4F7"),
    borderPadding=6,
    spaceBefore=5,
    spaceAfter=9,
))

# ---------------------------------------------------------------------
# Markdown inline formatting
# ---------------------------------------------------------------------

def inline_markup(text):
    text = escape(text)

    # Bold
    text = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        text
    )

    # Italic
    text = re.sub(
        r"(?<!\*)\*([^*]+?)\*(?!\*)",
        r"<i>\1</i>",
        text
    )

    # Inline code
    text = re.sub(
        r"`(.+?)`",
        r"<font name='Courier'>\1</font>",
        text
    )

    return text


# ---------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------

def parse_table(lines):
    rows = []

    for line in lines:

        # Skip Markdown table separator rows.
        if re.match(r"^\|?[\s:|-]+\|?$", line.strip()):
            continue

        cells = [
            cell.strip()
            for cell in line.strip().strip("|").split("|")
        ]

        rows.append([
            Paragraph(
                inline_markup(cell),
                styles["TableCell"]
            )
            for cell in cells
        ])

    if not rows:
        return []

    column_count = len(rows[0])
    available_width = 170 * mm
    column_width = available_width / column_count

    table = Table(
        rows,
        colWidths=[column_width] * column_count,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#DCE6F1"),
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.HexColor("#17365D"),
        ),
        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold",
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.35,
            colors.HexColor("#AAB7C4"),
        ),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    return [table, Spacer(1, 8)]


# ---------------------------------------------------------------------
# Mathematical equation rendering
# ---------------------------------------------------------------------

def render_equation(latex, output_path):
    """
    Render a LaTeX mathematical expression as a transparent PNG.
    Matplotlib MathText handles the equation formatting.
    """

    latex = latex.strip()

    # Remove display-math delimiters if present.
    if latex.startswith(r"\[") and latex.endswith(r"\]"):
        latex = latex[2:-2].strip()

    elif latex.startswith(r"\(") and latex.endswith(r"\)"):
        latex = latex[2:-2].strip()

    fig = plt.figure(figsize=(8, 0.8))
    fig.patch.set_alpha(0)

    fig.text(
        0.5,
        0.5,
        f"${latex}$",
        ha="center",
        va="center",
        fontsize=13,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.08,
    )

    plt.close(fig)


def build_equation_image(latex, equation_number):
    """
    Create a ReportLab image while preserving the equation's aspect ratio.
    """

    EQUATION_DIR.mkdir(parents=True, exist_ok=True)

    equation_path = (
        EQUATION_DIR / f"equation_{equation_number}.png"
    )

    render_equation(latex, equation_path)

    equation_image = Image(str(equation_path))

    max_width = 160 * mm
    max_height = 22 * mm

    scale = min(
        max_width / equation_image.imageWidth,
        max_height / equation_image.imageHeight,
        1.0,
    )

    equation_image.drawWidth = equation_image.imageWidth * scale
    equation_image.drawHeight = equation_image.imageHeight * scale
    equation_image.hAlign = "CENTER"

    return equation_image


# ---------------------------------------------------------------------
# Page footer
# ---------------------------------------------------------------------

def footer(canvas, doc):
    canvas.saveState()

    canvas.setStrokeColor(
        colors.HexColor("#D5DDE5")
    )

    canvas.line(
        20 * mm,
        15 * mm,
        190 * mm,
        15 * mm,
    )

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(
        colors.HexColor("#64748B")
    )

    canvas.drawString(
        20 * mm,
        10 * mm,
        "Temporal and Multimodal Visual Intelligence",
    )

    canvas.drawRightString(
        190 * mm,
        10 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


# ---------------------------------------------------------------------
# Main PDF generation
# ---------------------------------------------------------------------

def main():

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Report not found: {SOURCE}"
        )

    lines = SOURCE.read_text(
        encoding="utf-8"
    ).splitlines()

    story = []

    i = 0
    equation_number = 0

    while i < len(lines):

        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # -------------------------------------------------------------
        # Markdown tables
        # -------------------------------------------------------------

        if line.startswith("|"):

            table_lines = []

            while (
                i < len(lines)
                and lines[i].strip().startswith("|")
            ):
                table_lines.append(lines[i].strip())
                i += 1

            story.extend(parse_table(table_lines))
            continue

        # -------------------------------------------------------------
        # Code blocks
        # -------------------------------------------------------------

        if line.startswith("```"):

            code_lines = []
            i += 1

            while (
                i < len(lines)
                and not lines[i].strip().startswith("```")
            ):
                code_lines.append(lines[i])
                i += 1

            code_content = "<br/>".join(
                escape(code_line)
                for code_line in code_lines
            )

            story.append(
                Paragraph(
                    code_content,
                    styles["CodeBlock"]
                )
            )

            i += 1
            continue

        # -------------------------------------------------------------
        # Mathematical equations
        # -------------------------------------------------------------

        if line.startswith(r"\[") or line.startswith(r"\("):

            equation_lines = []

            while i < len(lines):

                equation_line = lines[i].strip()
                equation_lines.append(equation_line)

                if (
                    equation_line.endswith(r"\]")
                    or equation_line.endswith(r"\)")
                ):
                    break

                i += 1

            equation_text = " ".join(equation_lines)

            equation_number += 1

            story.append(
                build_equation_image(
                    equation_text,
                    equation_number,
                )
            )

            story.append(Spacer(1, 5))

            i += 1
            continue

        # -------------------------------------------------------------
        # Headings
        # -------------------------------------------------------------

        if line.startswith("# "):

            title = line[2:]

            story.append(
                Paragraph(
                    escape(title),
                    styles["ReportTitle"]
                )
            )

            story.append(Spacer(1, 8))

        elif line.startswith("## "):

            heading = line[3:]

            story.append(
                Paragraph(
                    escape(heading),
                    styles["Section"]
                )
            )

        elif line.startswith("### "):

            heading = line[4:]

            story.append(
                Paragraph(
                    escape(heading),
                    styles["Subsection"]
                )
            )

        # -------------------------------------------------------------
        # Ordered and unordered lists
        # -------------------------------------------------------------

        elif (
            line.startswith("- ")
            or re.match(r"^\d+\.\s", line)
        ):

            content = re.sub(
                r"^(?:-|\d+\.)\s+",
                "",
                line
            )

            story.append(
                Paragraph(
                    "• " + inline_markup(content),
                    styles["BodyReport"]
                )
            )

        # -------------------------------------------------------------
        # Regular paragraphs
        # -------------------------------------------------------------

        else:

            story.append(
                Paragraph(
                    inline_markup(line),
                    styles["BodyReport"]
                )
            )

        i += 1

    # -----------------------------------------------------------------
    # Build PDF
    # -----------------------------------------------------------------

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=23 * mm,
        title="Temporal and Multimodal Visual Intelligence",
        author="Research Report",
    )

    doc.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
    )

    print(f"PDF created successfully: {OUTPUT}")
    print(f"Equations rendered: {equation_number}")


if __name__ == "__main__":
    main()
