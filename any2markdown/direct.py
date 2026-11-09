"""Direct (non-OCR) extraction of Markdown from text-based documents.

The conversion pipeline is model-first: files are rendered to page images and
sent to a vision model. That is necessary for scans, photos and legacy binary
formats, but wasteful for files that already carry machine-readable text.

``try_direct_parse`` handles those files deterministically and returns one
``DirectPage`` per logical page. A page whose ``markdown`` is ``None`` could
not be extracted and still needs the OCR path (e.g. a scanned page inside an
otherwise text-based PDF).
"""

from __future__ import annotations

import contextlib
import io
import logging
import multiprocessing
import os
import re
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from any2markdown.models import FileType

# pymupdf4llm triggers this recommendation on import and PyMuPDF prints it to
# stdout, which would corrupt the CLI's markdown output.
os.environ.setdefault("PYMUPDF_SUGGEST_LAYOUT_ANALYZER", "0")

logger = logging.getLogger(__name__)

_pymupdf_messages_configured = False
# pymupdf4llm 2.x uses a process-global stdout redirect target for some
# status output, so PDF parsing is serialized while that redirect is active.
_pdf_parse_lock = threading.Lock()

# Text-layer PDFs with at least this many pages are parsed in a process pool
# once start_pdf_pool() has been called (see _pdf_to_markdown); smaller ones
# are not worth the hand-off.
_MIN_PAGES_FOR_POOL = 4
_pdf_workers = 0
_pdf_pool: ProcessPoolExecutor | None = None
_pdf_pool_lock = threading.Lock()

# Below these thresholds a page/document is treated as having no usable text
# layer and is handed to the vision model instead. Office formats have no
# invisible text layer, so any body text is enough to prefer direct parsing.
MIN_PAGE_CHARS = 20
MIN_PDF_DOC_CHARS = 40
MIN_OFFICE_CHARS = 1

_PICTURE_PLACEHOLDER = re.compile(
    r"^\s*\*\*==>\s*picture\b.*?<==\*\*\s*$", re.MULTILINE
)
_HEADING_STYLE = re.compile(r"^(?:heading|标题)\s*([1-6])$", re.IGNORECASE)


@dataclass
class DirectPage:
    """One logical page produced without OCR.

    ``markdown is None`` means this page has no usable text layer and must be
    rendered and sent to the model.
    """

    page_index: int
    markdown: str | None


def try_direct_parse(file_path: Path, file_type: FileType) -> list[DirectPage] | None:
    """Parse ``file_path`` without OCR.

    Returns one entry per logical page, or ``None`` when the file cannot be
    parsed directly (scanned PDF, image-only Office document, legacy format,
    missing optional dependency, or a parse error).
    """
    parser = _PARSERS.get(file_type)
    if parser is None:
        return None

    try:
        return parser(file_path)
    except ImportError as exc:
        logger.warning(
            "Direct parser for %s unavailable (%s); falling back to OCR",
            file_type.value,
            exc,
        )
        return None
    except Exception:
        logger.exception("Direct parse failed for %s; falling back to OCR", file_path)
        return None


# ---- PDF ---------------------------------------------------------------------


def _configure_pymupdf_messages() -> None:
    """Keep PyMuPDF 2.x status messages out of the CLI's stdout.

    PyMuPDF 2.x bundles pymupdf4llm and prints messages such as
    ``=== Document parser messages ===`` to stdout. Route them to the
    ``pymupdf`` logger at DEBUG so the Markdown written to stdout stays clean.
    PyMuPDF 1.x has no ``set_messages``; the guard keeps that path untouched.
    """
    global _pymupdf_messages_configured
    if _pymupdf_messages_configured:
        return
    _pymupdf_messages_configured = True

    try:
        import pymupdf
    except ImportError:
        return

    set_messages = getattr(pymupdf, "set_messages", None)
    if set_messages is None:
        return

    try:
        set_messages(
            pylogging=True,
            pylogging_name="pymupdf",
            pylogging_level=logging.DEBUG,
        )
    except Exception:
        logger.debug("Could not redirect PyMuPDF messages", exc_info=True)


def _parse_pdf(file_path: Path) -> list[DirectPage] | None:
    try:
        import pymupdf as fitz
    except ImportError:  # older releases only expose fitz
        import fitz

    with fitz.open(file_path) as doc:
        page_count = len(doc)
        raw_text: list[str] = []
        scanned: list[bool] = []
        for index in range(page_count):
            page = doc[index]
            text = page.get_text("text")
            raw_text.append(text)
            chars = _meaningful_len(text)
            scanned.append(
                chars >= MIN_PAGE_CHARS and _looks_scanned(page, chars)
            )

    if sum(_meaningful_len(text) for text in raw_text) < MIN_PDF_DOC_CHARS:
        return None

    text_pages = [
        i
        for i, text in enumerate(raw_text)
        if _meaningful_len(text) >= MIN_PAGE_CHARS and not scanned[i]
    ]
    if not text_pages:
        return None

    parsed = dict(_pdf_to_markdown(str(file_path), text_pages))
    return [DirectPage(i, parsed.get(i) or None) for i in range(page_count)]


def default_pdf_workers() -> int:
    """min(usable CPUs, 4)."""
    try:
        cpus = len(os.sched_getaffinity(0))  # respects container CPU pinning
    except AttributeError:
        cpus = os.cpu_count() or 1
    return min(cpus, 4)


def start_pdf_pool(workers: int) -> None:
    """Parse text-layer PDFs in ``workers`` processes from now on.

    Meant for long-running processes (the HTTP server). Blocks until every
    worker has started and imported pymupdf4llm — several seconds that would
    otherwise land on the first request — so call it from a background thread.
    Without this call, or with ``workers <= 1``, PDFs are parsed in-process.
    """
    global _pdf_workers
    if workers <= 1:
        return
    _pdf_workers = workers
    pool = _get_pdf_pool(workers)
    # Each submit while no worker is idle starts a new process, so this
    # brings up the whole pool.
    for f in [pool.submit(_warm_worker) for _ in range(workers)]:
        f.result()
    logger.info("PDF parsing pool ready: %d worker process(es)", workers)


def _warm_worker() -> None:
    import pymupdf4llm  # noqa: F401  (the slow part of a worker's start-up)

    _configure_pymupdf_messages()


def _get_pdf_pool(workers: int) -> ProcessPoolExecutor:
    global _pdf_pool
    with _pdf_pool_lock:
        if _pdf_pool is None:
            # spawn, not fork: the server process has threads (event loop,
            # thread pool) and forking those is unsafe.
            _pdf_pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context("spawn")
            )
        return _pdf_pool


def _reset_pdf_pool() -> None:
    global _pdf_pool
    with _pdf_pool_lock:
        if _pdf_pool is not None:
            _pdf_pool.shutdown(wait=False, cancel_futures=True)
            _pdf_pool = None


def _pdf_to_markdown(path: str, pages: list[int]) -> list[tuple[int, str]]:
    """Run pymupdf4llm on ``pages``, spread over worker processes when worthwhile.

    pymupdf4llm is pure-Python heavy (table detection, layout analysis), so
    threads would not help; separate processes do.
    """
    workers = _pdf_workers
    if workers <= 1 or len(pages) < _MIN_PAGES_FOR_POOL:
        return _pdf_pages_to_markdown(path, pages)

    # Several small chunks per worker so one slow page (a dense figure, a big
    # table) does not leave the other workers idle. Header levels stay
    # consistent across chunks: pymupdf4llm derives them from the whole
    # document, not just the pages it is asked for.
    n_chunks = min(len(pages), workers * 3)
    size = -(-len(pages) // n_chunks)
    chunks = [pages[i : i + size] for i in range(0, len(pages), size)]
    try:
        pool = _get_pdf_pool(workers)
        futures = [pool.submit(_pdf_pages_to_markdown, path, c) for c in chunks]
        return [item for f in futures for item in f.result()]
    except BrokenProcessPool:
        logger.warning("PDF worker pool broke; parsing %s in-process", path)
        _reset_pdf_pool()
        return _pdf_pages_to_markdown(path, pages)


def _pdf_pages_to_markdown(path: str, pages: list[int]) -> list[tuple[int, str]]:
    """``(page_index, markdown)`` for ``pages``. Runs in a worker or in-process."""
    import pymupdf4llm

    _configure_pymupdf_messages()

    # pymupdf4llm 2.x writes a bare newline to stdout when it has status
    # messages. Send anything it prints to stderr so CLI markdown stays clean.
    stdout = sys.stderr if sys.stderr is not None else io.StringIO()
    with _pdf_parse_lock, contextlib.redirect_stdout(stdout):
        # ignore_images: we drop image placeholders anyway, and pymupdf4llm
        # hashes every image in the page's resources to locate it. Pages that
        # share a large resource dictionary (seen: 3,654 listed, 12 drawn)
        # spent ~15s each on that.
        chunks = pymupdf4llm.to_markdown(
            path, pages=pages, page_chunks=True, ignore_images=True
        )

    out: list[tuple[int, str]] = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        # PyMuPDF 1.x / pymupdf4llm 0.x use "page"; PyMuPDF 2.x uses
        # "page_number". Both are 1-based.
        page_number = metadata.get("page", metadata.get("page_number"))
        if page_number is None:
            continue
        out.append((int(page_number) - 1, _clean_pdf_markdown(chunk.get("text", ""))))
    return out


def _looks_scanned(page, text_chars: int) -> bool:
    """A page dominated by one image with only a little text is a scan.

    Scanned documents often carry a short header/footer text layer; without
    this check those pages would be "parsed" into just the header.
    """
    if text_chars >= 200:
        return False

    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return False

    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return False

    text_area = 0.0
    image_area = 0.0
    for block in blocks:
        bbox = block.get("bbox")
        if not bbox:
            continue
        area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        if block.get("type") == 1:  # image block
            image_area = max(image_area, area)
        else:
            text_area += area

    return image_area >= page_area * 0.1 and image_area > text_area * 3


def _clean_pdf_markdown(markdown: str) -> str:
    # pymupdf4llm leaves placeholders for images it did not extract.
    return _PICTURE_PLACEHOLDER.sub("", markdown).strip()


# ---- DOCX --------------------------------------------------------------------


def _parse_docx(file_path: Path) -> list[DirectPage] | None:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(file_path))

    parts: list[str] = []
    chars = 0
    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            markdown = _docx_paragraph_to_md(block)
        elif isinstance(block, Table):
            markdown = _docx_table_to_md(block)
        else:
            continue
        if markdown:
            parts.append(markdown)
            chars += _meaningful_len(markdown)

    if chars < MIN_OFFICE_CHARS:
        return None
    return [DirectPage(0, "\n\n".join(parts))]


def _iter_docx_blocks(document):
    """Yield top-level paragraphs and tables in document order."""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    if hasattr(document, "iter_inner_content"):
        yield from document.iter_inner_content()
        return

    # python-docx < 1.1 fallback.
    from docx.oxml.ns import qn

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _docx_paragraph_to_md(paragraph) -> str:
    text = _docx_inline_to_md(paragraph).strip()
    if not text:
        return ""

    style = paragraph.style.name.strip() if paragraph.style is not None else ""
    lower = style.lower()

    match = _HEADING_STYLE.match(style)
    if match:
        return f"{'#' * int(match.group(1))} {text}"
    if lower in ("title", "subtitle"):
        return f"# {text}"
    if "list bullet" in lower:
        return f"- {text}"
    if "list number" in lower:
        return f"1. {text}"
    if "quote" in lower:
        return f"> {text}"
    return text


def _docx_inline_to_md(paragraph) -> str:
    """Render runs and hyperlinks in order.

    ``Paragraph.runs`` omits runs nested inside ``w:hyperlink``, so using it
    alone would silently drop link text.
    """
    items = (
        paragraph.iter_inner_content()
        if hasattr(paragraph, "iter_inner_content")
        else paragraph.runs
    )

    out: list[str] = []
    for item in items:
        if hasattr(item, "url"):  # Hyperlink
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            out.append(f"[{text}]({item.url})" if item.url else text)
        else:
            out.append(_docx_run_to_md(item))
    return "".join(out)


def _docx_run_to_md(run) -> str:
    text = run.text
    if not text:
        return ""

    lead = text[: len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()) :]
    core = text.strip()
    if core:
        if run.font is not None and run.font.strike:
            core = f"~~{core}~~"
        bold = bool(run.bold)
        italic = bool(run.italic)
        if bold and italic:
            core = f"***{core}***"
        elif bold:
            core = f"**{core}**"
        elif italic:
            core = f"*{core}*"
    return f"{lead}{core}{trail}"


def _docx_table_to_md(table) -> str:
    rows = [
        [_docx_cell_text(cell) for cell in row.cells] for row in table.rows
    ]
    return _md_table(rows)


def _docx_cell_text(cell) -> str:
    parts = [p.text.strip() for p in cell.paragraphs if p.text.strip()]
    return _escape_cell("\n".join(parts))


# ---- PPTX --------------------------------------------------------------------


def _parse_pptx(file_path: Path) -> list[DirectPage] | None:
    from pptx import Presentation

    presentation = Presentation(str(file_path))

    pages: list[DirectPage] = []
    total_chars = 0

    for index, slide in enumerate(presentation.slides):
        parts: list[str] = []
        title_shape = slide.shapes.title
        title_id = title_shape.shape_id if title_shape is not None else None

        if title_shape is not None and title_shape.has_text_frame:
            title = " ".join(title_shape.text_frame.text.split())
            if title:
                parts.append(f"## {title}")
                total_chars += _meaningful_len(title)

        for shape in slide.shapes:
            if shape.shape_id == title_id:
                continue

            if getattr(shape, "has_text_frame", False):
                for paragraph in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in paragraph.runs).strip()
                    if not text:
                        continue
                    total_chars += _meaningful_len(text)
                    level = max(0, int(getattr(paragraph, "level", 0) or 0))
                    parts.append(f"{'  ' * level}- {text}")

            if getattr(shape, "has_table", False):
                table_md = _pptx_table_to_md(shape.table)
                if table_md:
                    parts.append(table_md)
                    total_chars += _meaningful_len(table_md)

        if not parts:
            parts.append(f"<!-- slide {index + 1}: no extractable text -->")
        pages.append(DirectPage(index, "\n\n".join(parts)))

    if total_chars < MIN_OFFICE_CHARS:
        return None
    return pages


def _pptx_table_to_md(table) -> str:
    rows = [
        [_escape_cell(cell.text.strip()) for cell in row.cells]
        for row in table.rows
    ]
    return _md_table(rows)


# ---- XLSX --------------------------------------------------------------------


def _parse_xlsx(file_path: Path) -> list[DirectPage] | None:
    from openpyxl import load_workbook

    workbook = load_workbook(str(file_path), read_only=True, data_only=True)

    pages: list[DirectPage] = []
    total_chars = 0
    try:
        for index, sheet in enumerate(workbook.worksheets):
            rows: list[list[str]] = []
            for values in sheet.iter_rows(values_only=True):
                cells = [_format_cell(value) for value in values]
                while cells and cells[-1] == "":
                    cells.pop()
                if any(cells):
                    rows.append(cells)

            if not rows:
                pages.append(DirectPage(index, ""))
                continue

            markdown = f"## {sheet.title}\n\n{_md_table(rows)}"
            pages.append(DirectPage(index, markdown))
            total_chars += _meaningful_len(markdown)
    finally:
        workbook.close()

    if total_chars < MIN_OFFICE_CHARS:
        return None
    return pages


def _format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return _escape_cell(str(value))


# ---- Markdown helpers --------------------------------------------------------


def _md_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return ""

    normalized = [row + [""] * (width - len(row)) for row in rows]
    header, body = normalized[0], normalized[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _escape_cell(text: str) -> str:
    return (
        text.replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\n", "<br>")
        .strip()
    )


def _meaningful_len(text: str) -> int:
    return sum(
        1
        for char in text
        if not char.isspace()
        and char.isprintable()
        and char != chr(0xFFFD)  # undecodable glyph from a bad text layer
    )


_PARSERS = {
    FileType.PDF: _parse_pdf,
    FileType.DOCX: _parse_docx,
    FileType.PPTX: _parse_pptx,
    FileType.XLSX: _parse_xlsx,
}
