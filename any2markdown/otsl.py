"""OTSL → HTML table conversion.

The vision model emits tables in OTSL (Omni-Table Structure Language) v1.0,
which uses six tags:

    <fcel>  filled cell (carries text)        <ecel>  empty cell
    <nl>    new row                           <lcel>  merge-right (colspan continuation)
    <ucel>  merge-down (rowspan continuation) <xcel>  merge-right-and-down

``render_tables(text, table_format)`` is the high-level entry used by the
pipeline: it finds OTSL table runs in a page's model markdown and, when
``table_format == "html"``, converts each to an HTML ``<table>``; ``"otsl"``
leaves them untouched. Conversion errors are non-fatal — a failed table is
returned as-is so a page is never lost because of one bad table.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import List

OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"

_ALL_TAGS = [OTSL_FCEL, OTSL_ECEL, OTSL_NL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]
_TAG_ALT = "|".join(_ALL_TAGS)

_NON_CAPTURING_TAG_GROUP = f"(?:{_TAG_ALT})"
OTSL_FIND_PATTERN = re.compile(
    f"{_NON_CAPTURING_TAG_GROUP}.*?(?={_NON_CAPTURING_TAG_GROUP}|$)",
    flags=re.DOTALL,
)

# A real table always contains at least one filled/empty cell; used to avoid
# treating a stray <nl> in prose as a table.
_CELL_TAG_RE = re.compile(f"{OTSL_FCEL}|{OTSL_ECEL}")
_TAG_RE = re.compile(_TAG_ALT)


@dataclass
class TableCell:
    """A single table cell with span + grid-index metadata."""

    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int = 0
    end_row_offset_idx: int = 0
    start_col_offset_idx: int = 0
    end_col_offset_idx: int = 0
    text: str = ""
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False


@dataclass
class TableData:
    """A table: its cells plus row/column counts, with a 2D ``grid`` view."""

    table_cells: List[TableCell] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0

    @property
    def grid(self) -> List[List[TableCell]]:
        table_data = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]
        for cell in self.table_cells:
            for i in range(
                min(cell.start_row_offset_idx, self.num_rows),
                min(cell.end_row_offset_idx, self.num_rows),
            ):
                for j in range(
                    min(cell.start_col_offset_idx, self.num_cols),
                    min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[i][j] = cell
        return table_data


def otsl_extract_tokens_and_text(s: str):
    """Split an OTSL string into (tags, interleaved tags+text)."""
    pattern = r"(" + "|".join(_ALL_TAGS) + r")"
    tokens = re.findall(pattern, s)
    text_parts = re.split(pattern, s)
    text_parts = [token for token in text_parts if token.strip()]
    return tokens, text_parts


def otsl_parse_texts(texts, tokens):
    """Parse OTSL tokens + interleaved text into TableCell objects and rows."""
    split_word = OTSL_NL
    split_row_tokens = [
        list(y)
        for x, y in itertools.groupby(tokens, lambda z: z == split_word)
        if not x
    ]
    table_cells = []
    r_idx = 0
    c_idx = 0

    # Ensure matrix completeness
    if split_row_tokens:
        max_cols = max(len(row) for row in split_row_tokens)
        for row in split_row_tokens:
            while len(row) < max_cols:
                row.append(OTSL_ECEL)
        new_texts = []
        text_idx = 0
        for row in split_row_tokens:
            for token in row:
                new_texts.append(token)
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    if text_idx < len(texts) and texts[text_idx] not in _ALL_TAGS:
                        new_texts.append(texts[text_idx])
                        text_idx += 1
            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1
        texts = new_texts

    def count_right(tokens, c_idx, r_idx, which_tokens):
        span = 0
        c_idx_iter = c_idx
        while tokens[r_idx][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(tokens, c_idx, r_idx, which_tokens):
        span = 0
        r_idx_iter = r_idx
        while tokens[r_idx_iter][c_idx] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in [OTSL_FCEL, OTSL_ECEL]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL:
                cell_text = texts[i + 1]
                right_offset = 2

            next_right_cell = (
                texts[i + right_offset] if i + right_offset < len(texts) else ""
            )
            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens):
                if c_idx < len(split_row_tokens[r_idx + 1]):
                    next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

            if next_right_cell in [OTSL_LCEL, OTSL_XCEL]:
                col_span += count_right(
                    split_row_tokens, c_idx + 1, r_idx, [OTSL_LCEL, OTSL_XCEL]
                )
            if next_bottom_cell in [OTSL_UCEL, OTSL_XCEL]:
                row_span += count_down(
                    split_row_tokens, c_idx, r_idx + 1, [OTSL_UCEL, OTSL_XCEL]
                )

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )
        if text in [OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0
    return table_cells, split_row_tokens


def export_to_html(table_data: TableData) -> str:
    """Render a TableData to an HTML ``<table>`` string."""
    nrows = table_data.num_rows
    ncols = table_data.num_cols
    if len(table_data.table_cells) == 0:
        return ""
    body = ""
    grid = table_data.grid
    for i in range(nrows):
        body += "<tr>"
        for j in range(ncols):
            cell: TableCell = grid[i][j]
            rowspan, rowstart = (cell.row_span, cell.start_row_offset_idx)
            colspan, colstart = (cell.col_span, cell.start_col_offset_idx)
            if rowstart != i or colstart != j:
                continue
            content = cell.text.strip()
            celltag = "th" if cell.column_header else "td"
            opening_tag = f"{celltag}"
            if rowspan > 1:
                opening_tag += f' rowspan="{rowspan}"'
            if colspan > 1:
                opening_tag += f' colspan="{colspan}"'
            body += f"<{opening_tag}>{content}</{celltag}>"
        body += "</tr>"
    body = f"<table>{body}</table>"
    return body


def otsl_pad_to_sqr_v2(otsl_str: str) -> str:
    """Pad an OTSL string so every row has the same number of cells."""
    assert isinstance(otsl_str, str)
    otsl_str = otsl_str.strip()
    if OTSL_NL not in otsl_str:
        return otsl_str + OTSL_NL
    lines = otsl_str.split(OTSL_NL)
    row_data = []
    for line in lines:
        if not line:
            continue
        raw_cells = OTSL_FIND_PATTERN.findall(line)
        if not raw_cells:
            continue
        total_len = len(raw_cells)
        min_len = 0
        for i, cell_str in enumerate(raw_cells):
            if cell_str.startswith(OTSL_FCEL):
                min_len = i + 1
        row_data.append(
            {"raw_cells": raw_cells, "total_len": total_len, "min_len": min_len}
        )
    if not row_data:
        return OTSL_NL
    global_min_width = max(row["min_len"] for row in row_data) if row_data else 0
    max_total_len = max(row["total_len"] for row in row_data) if row_data else 0
    search_start = global_min_width
    search_end = max(global_min_width, max_total_len)
    min_total_cost = float("inf")
    optimal_width = search_end

    for width in range(search_start, search_end + 1):
        current_total_cost = sum(abs(row["total_len"] - width) for row in row_data)
        if current_total_cost < min_total_cost:
            min_total_cost = current_total_cost
            optimal_width = width

    repaired_lines = []
    for row in row_data:
        cells = row["raw_cells"]
        current_len = len(cells)
        if current_len > optimal_width:
            new_cells = cells[:optimal_width]
        else:
            padding = [OTSL_ECEL] * (optimal_width - current_len)
            new_cells = cells + padding
        repaired_lines.append("".join(new_cells))
    return OTSL_NL.join(repaired_lines) + OTSL_NL


def convert_otsl_to_html(otsl_content: str) -> str:
    """Convert an OTSL v1.0 string to an HTML ``<table>``.

    Only the six OTSL tags are recognized: ``<fcel> <ecel> <nl> <lcel> <ucel> <xcel>``.
    """
    otsl_content = otsl_pad_to_sqr_v2(otsl_content)
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)
    table_data = TableData(
        num_rows=len(split_row_tokens),
        num_cols=(max(len(row) for row in split_row_tokens) if split_row_tokens else 0),
        table_cells=table_cells,
    )
    return export_to_html(table_data)


def render_tables(text: str, table_format: str = "html") -> str:
    """Convert OTSL table runs in ``text`` per ``table_format``.

    - ``"otsl"``: return ``text`` unchanged (model's OTSL kept as-is).
    - ``"html"``: find OTSL table runs (both ```` ```otsl ````-fenced and bare)
      and convert each to an HTML ``<table>``.

    Any single-table conversion error is non-fatal: the original OTSL block is
    kept so the rest of the page is unaffected.
    """
    if table_format == "otsl":
        return text
    if table_format != "html":
        raise ValueError(
            f"unknown table_format {table_format!r}; expected 'html' or 'otsl'"
        )
    if not text or _TAG_RE.search(text) is None:
        return text

    def _convert_block(block: str) -> str:
        try:
            html = convert_otsl_to_html(block.strip())
            return html if html else block
        except Exception:
            return block

    # 1) Fenced code blocks whose content is OTSL → replace whole fence.
    def _fence_repl(m):
        body = m.group(2)
        if _CELL_TAG_RE.search(body):
            return _convert_block(body)
        return m.group(0)

    text = re.sub(r"```([A-Za-z0-9_-]*)\n(.*?)```", _fence_repl, text, flags=re.DOTALL)
    if _TAG_RE.search(text) is None:
        return text

    # 2) Bare OTSL runs: group consecutive lines that contain OTSL tags. The
    #    model emits a table as contiguous OTSL lines, so a non-OTSL line ends a
    #    run and is preserved verbatim. Within a run, only the span from the
    #    first tag to the last tag is the table; leading text on the first line
    #    and trailing text after a structural tag (<nl>/<lcel>/<ucel>/<xcel>)
    #    are kept as prose (a table that ends on a cell tag absorbs trailing
    #    text as cell content, matching pure-OTSL semantics).
    _STRUCT_TAGS = (OTSL_NL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL)

    def _convert_bare_run(run: str) -> str:
        first = _TAG_RE.search(run)
        if first is None:
            return run
        last_m = first
        for m in _TAG_RE.finditer(run):
            last_m = m
        leading = run[: first.start()]
        last_tag = last_m.group()
        if last_tag in _STRUCT_TAGS:
            otsl = run[first.start(): last_m.end()]
            trailing = run[last_m.end():]
        else:  # ends on a cell tag (<fcel>/<ecel>): trailing text is cell content
            otsl = run[first.start():]
            trailing = ""
        return f"{leading}{_convert_block(otsl)}{trailing}"

    lines = text.split("\n")
    out_lines: list[str] = []
    i = 0
    while i < len(lines):
        if _TAG_RE.search(lines[i]):
            j = i
            while j < len(lines) and _TAG_RE.search(lines[j]):
                j += 1
            out_lines.append(_convert_bare_run("\n".join(lines[i:j])))
            i = j
        else:
            out_lines.append(lines[i])
            i += 1
    return "\n".join(out_lines)
