#!/usr/bin/env python3
"""Render a compact solved-theorem rate table from a small JSON summary."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

BACKGROUND = (249, 250, 252)
PAPER = (255, 255, 255)
INK = (27, 36, 52)
MUTED = (86, 99, 119)
GRID = (205, 213, 224)
TRACK = (225, 231, 239)
BLUE = (42, 105, 176)


def font_path(bold: bool = False) -> str:
    """Return a common sans-serif font available on macOS or Linux."""
    candidates = [
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Verdana.ttf",
        "/Library/Fonts/Verdana Bold.ttf" if bold else "/Library/Fonts/Verdana.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("No supported Verdana or DejaVu Sans font found")


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path(bold), size=size)


def wrap_pixels(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
) -> list[str]:
    """Wrap text by measured pixel width."""
    approximate_chars = max(12, int(max_width / max(font.size * 0.55, 1)))
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        for candidate in textwrap.wrap(paragraph, width=approximate_chars) or [""]:
            words = candidate.split()
            current = ""
            for word in words:
                trial = f"{current} {word}".strip()
                if draw.textlength(trial, font=font) <= max_width or not current:
                    current = trial
                else:
                    lines.append(current)
                    current = word
            if current:
                lines.append(current)
    return lines


def blend(
    start: tuple[int, int, int], end: tuple[int, int, int], amount: float
) -> tuple[int, int, int]:
    amount = max(0.0, min(1.0, amount))
    return tuple(round(a + (b - a) * amount) for a, b in zip(start, end, strict=True))


def format_rate(solved: int, total: int) -> str:
    rate = 100 * solved / total
    if rate == 0 or rate == 100 or abs(rate - round(rate)) < 0.05:
        return f"{rate:.0f}%"
    return f"{rate:.1f}%"


def validate(data: dict[str, Any]) -> None:
    required = {"title", "subtitle", "note", "columns", "rows"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")
    keys = [column["key"] for column in data["columns"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Column keys must be unique")
    for column in data["columns"]:
        if not isinstance(column["total"], int) or column["total"] <= 0:
            raise ValueError(f"Invalid denominator for {column['key']}")
    for row in data["rows"]:
        if set(row["values"]) != set(keys):
            raise ValueError(f"Row {row['label']} does not define every column")
        for column in data["columns"]:
            solved = row["values"][column["key"]]
            if not isinstance(solved, int) or not 0 <= solved <= column["total"]:
                raise ValueError(f"Invalid solved count for {row['label']} / {column['key']}")


def render(data: dict[str, Any], output: Path) -> None:
    validate(data)
    columns = data["columns"]
    rows = data["rows"]
    n_columns = len(columns)

    margin_x = 58
    label_width = 350 if max(len(row["label"]) for row in rows) > 24 else 300
    cell_width = 112 if n_columns >= 10 else 220 if n_columns >= 5 else 245
    row_height = 92
    header_height = 88
    title_font = load_font(34, bold=True)
    subtitle_font = load_font(19)
    note_font = load_font(15)
    header_font = load_font(17, bold=True)
    header_small_font = load_font(13)
    label_font = load_font(17)
    label_bold_font = load_font(17, bold=True)
    rate_font = load_font(19 if cell_width >= 180 else 17, bold=True)
    count_font = load_font(14 if cell_width >= 180 else 13)

    width = margin_x * 2 + label_width + cell_width * n_columns
    scratch = Image.new("RGB", (width, 400), BACKGROUND)
    scratch_draw = ImageDraw.Draw(scratch)
    subtitle_lines = wrap_pixels(
        scratch_draw, data["subtitle"], subtitle_font, width - margin_x * 2
    )
    note_lines = wrap_pixels(scratch_draw, data["note"], note_font, width - margin_x * 2)
    title_height = 58 + len(subtitle_lines) * 26 + 22
    footer_height = 34 + len(note_lines) * 21 + 28
    height = title_height + header_height + row_height * len(rows) + footer_height

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((margin_x, 30), data["title"], font=title_font, fill=INK)
    subtitle_y = 78
    for line in subtitle_lines:
        draw.text((margin_x, subtitle_y), line, font=subtitle_font, fill=MUTED)
        subtitle_y += 26

    table_left = margin_x
    grid_left = table_left + label_width
    header_top = title_height
    table_right = grid_left + cell_width * n_columns
    table_bottom = header_top + header_height + row_height * len(rows)

    draw.rectangle(
        (table_left, header_top, table_right, table_bottom), fill=PAPER, outline=GRID, width=1
    )
    draw.text((table_left + 12, header_top + 29), "Model / condition", font=header_font, fill=INK)
    for index, column in enumerate(columns):
        x0 = grid_left + index * cell_width
        if column.get("emphasis"):
            draw.line((x0, header_top, x0, table_bottom), fill=MUTED, width=2)
        label = column["label"]
        label_width_px = draw.textlength(label, font=header_font)
        draw.text(
            (x0 + (cell_width - label_width_px) / 2, header_top + 18),
            label,
            font=header_font,
            fill=INK,
        )
        n_label = f"n={column['total']}"
        n_width = draw.textlength(n_label, font=header_small_font)
        draw.text(
            (x0 + (cell_width - n_width) / 2, header_top + 48),
            n_label,
            font=header_small_font,
            fill=MUTED,
        )

    for row_index, row in enumerate(rows):
        y0 = header_top + header_height + row_index * row_height
        if row.get("separator_before"):
            draw.line((table_left, y0, table_right, y0), fill=MUTED, width=2)
        else:
            draw.line((table_left, y0, table_right, y0), fill=GRID, width=1)
        label_y = y0 + (row_height - label_font.size) / 2 - 3
        draw.text(
            (table_left + 12, label_y),
            row["label"],
            font=label_bold_font if row.get("emphasis") else label_font,
            fill=INK,
        )
        for column_index, column in enumerate(columns):
            x0 = grid_left + column_index * cell_width
            solved = row["values"][column["key"]]
            total = column["total"]
            rate = solved / total
            cell_fill = blend(PAPER, BLUE, 0.05 + 0.20 * rate if rate else 0.0)
            draw.rectangle(
                (x0 + 1, y0 + 1, x0 + cell_width - 1, y0 + row_height - 1), fill=cell_fill
            )
            rate_label = format_rate(solved, total)
            rate_width = draw.textlength(rate_label, font=rate_font)
            draw.text(
                (x0 + (cell_width - rate_width) / 2, y0 + 18), rate_label, font=rate_font, fill=INK
            )
            count_label = f"{solved}/{total}"
            count_width = draw.textlength(count_label, font=count_font)
            draw.text(
                (x0 + (cell_width - count_width) / 2, y0 + 46),
                count_label,
                font=count_font,
                fill=MUTED,
            )
            track_left = x0 + 10
            track_right = x0 + cell_width - 10
            track_top = y0 + row_height - 13
            draw.rectangle((track_left, track_top, track_right, track_top + 5), fill=TRACK)
            if solved:
                draw.rectangle(
                    (
                        track_left,
                        track_top,
                        track_left + round((track_right - track_left) * rate),
                        track_top + 5,
                    ),
                    fill=BLUE,
                )

    for index in range(n_columns + 1):
        x = grid_left + index * cell_width
        draw.line((x, header_top, x, table_bottom), fill=GRID, width=1)

    note_y = table_bottom + 24
    for line in note_lines:
        draw.text((margin_x, note_y), line, font=note_font, fill=MUTED)
        note_y += 21

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True, dpi=(144, 144))
    if output.stat().st_size > 1_000_000:
        output.unlink()
        raise ValueError("Rendered figure exceeds the 1 MB repository limit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Path to the summary JSON")
    parser.add_argument("--output", type=Path, help="PNG destination (defaults beside the JSON)")
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    output = args.output or args.data.with_suffix(".png")
    render(data, output)


if __name__ == "__main__":
    main()
