import asyncio
import logging
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from any2markdown.assembler import Assembler
from any2markdown.direct import DirectPage, try_direct_parse
from any2markdown.model_client import ModelClient
from any2markdown.models import ConversionResult, FileType, PageImage, PageResult
from any2markdown.pipeline import Pipeline
from any2markdown.renderer.base import BaseRenderer
from any2markdown.renderer.image import ImageRenderer
from any2markdown.renderer.office import OfficeRenderer
from any2markdown.renderer.pdf import PdfRenderer
from any2markdown.utils import (
    DirectParseError,
    UnsupportedFileTypeError,
    create_temp_dir,
    detect_file_type,
)

logger = logging.getLogger(__name__)

PARSE_MODES = ("auto", "ocr", "direct")


class Any2Markdown:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "http://localhost:8000/v1",
        model: str = "xiaomi-ocr-0",
        dpi: int = 200,
        concurrency: int = 4,
        max_retries: int = 3,
        timeout: int = 60,
        prompt: str | None = None,
        temperature: float | None = None,
        max_image_size: int = 4096,
        debug_dir: str | Path | None = None,
        table_format: str = "html",
        parse_mode: str = "auto",
    ):
        if parse_mode not in PARSE_MODES:
            raise ValueError(
                f"Invalid parse_mode: {parse_mode!r}; expected one of {PARSE_MODES}"
            )

        self.dpi = dpi
        self.concurrency = concurrency
        self.max_image_size = max_image_size
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.parse_mode = parse_mode

        self.model_client = ModelClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            timeout=timeout,
        )

        self.pipeline = Pipeline(
            model_client=self.model_client,
            concurrency=concurrency,
            max_retries=max_retries,
            table_format=table_format,
        )

        self.assembler = Assembler()

        self._renderers: dict[FileType, BaseRenderer] = {
            FileType.IMAGE: ImageRenderer(max_size=max_image_size),
            FileType.PDF: PdfRenderer(dpi=dpi),
            FileType.DOC: OfficeRenderer(dpi=dpi),
            FileType.DOCX: OfficeRenderer(dpi=dpi),
            FileType.PPT: OfficeRenderer(dpi=dpi),
            FileType.PPTX: OfficeRenderer(dpi=dpi),
            FileType.XLSX: OfficeRenderer(dpi=dpi),
        }

    def _get_file_debug_dir(self, file_path: Path) -> Path | None:
        if self.debug_dir is None:
            return None
        d = self.debug_dir / file_path.stem
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _accept_direct(
        self,
        file_path: Path,
        file_type: FileType,
        pages: list[DirectPage] | None,
        parse_mode: str,
    ) -> list[DirectPage] | None:
        """Validate direct-parse output, or fall back to OCR."""
        if pages is None or not any(p.markdown is not None for p in pages):
            if parse_mode == "direct":
                raise DirectParseError(
                    f"{file_path.name} has no usable text layer; "
                    "use parse_mode=auto or ocr"
                )
            return None

        # Only PDF supports a per-page mix of direct text and OCR. Office
        # documents are either parsed as a whole or handed to the model.
        if file_type is not FileType.PDF and any(
            p.markdown is None for p in pages
        ):
            if parse_mode == "direct":
                raise DirectParseError(
                    f"{file_path.name} cannot be fully parsed directly; "
                    "use parse_mode=auto or ocr"
                )
            return None

        logger.info(
            "Direct parse: %s (%d logical page(s))", file_path.name, len(pages)
        )
        return pages

    def _direct_pages(
        self, file_path: Path, file_type: FileType
    ) -> list[DirectPage] | None:
        if self.parse_mode == "ocr":
            return None
        return self._accept_direct(
            file_path, file_type, try_direct_parse(file_path, file_type), self.parse_mode
        )

    async def _direct_pages_async(
        self, file_path: Path, file_type: FileType, parse_mode: str
    ) -> list[DirectPage] | None:
        if parse_mode == "ocr":
            return None
        pages = await asyncio.to_thread(try_direct_parse, file_path, file_type)
        return self._accept_direct(file_path, file_type, pages, parse_mode)

    def _direct_result(
        self, file_path: Path, pages: list[DirectPage]
    ) -> ConversionResult:
        results = [
            PageResult(page_index=p.page_index, markdown=p.markdown or "")
            for p in pages
        ]
        return ConversionResult(
            markdown=self.assembler.assemble(results),
            pages=results,
            source_file=str(file_path),
            total_pages=len(pages),
            failed_pages=[],
            parse_mode_used="direct",
        )

    def _build_result(
        self,
        file_path: Path,
        ocr_results: list[PageResult],
        direct_pages: list[DirectPage] | None,
        rendered_count: int,
    ) -> ConversionResult:
        if direct_pages is not None:
            results = sorted(
                [
                    *ocr_results,
                    *(
                        PageResult(page_index=p.page_index, markdown=p.markdown or "")
                        for p in direct_pages
                        if p.markdown is not None
                    ),
                ],
                key=lambda r: r.page_index,
            )
            total_pages = len(direct_pages)
            used = "hybrid"
        else:
            results = ocr_results
            total_pages = rendered_count
            used = "ocr"

        return ConversionResult(
            markdown=self.assembler.assemble(results),
            pages=results,
            source_file=str(file_path),
            total_pages=total_pages,
            failed_pages=[r.page_index for r in results if not r.success],
            parse_mode_used=used,
        )

    def convert(self, file_path: str | Path) -> ConversionResult:
        """同步转换接口"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = detect_file_type(file_path)
        renderer = self._renderers.get(file_type)
        if renderer is None:
            raise UnsupportedFileTypeError(f"No renderer for type: {file_type}")

        self.pipeline.debug_dir = self._get_file_debug_dir(file_path)

        direct_pages = self._direct_pages(file_path, file_type)
        if direct_pages is not None and all(
            p.markdown is not None for p in direct_pages
        ):
            return self._direct_result(file_path, direct_pages)

        tmp_dir = create_temp_dir()
        try:
            render_kwargs: dict = {"dpi": self.dpi}
            if direct_pages is not None:
                # Mixed PDF: render only the pages without a text layer.
                render_kwargs["page_indices"] = [
                    p.page_index for p in direct_pages if p.markdown is None
                ]

            pages: list[PageImage] = renderer.render(
                file_path, tmp_dir, **render_kwargs
            )

            if not pages:
                if direct_pages is not None:
                    return self._direct_result(file_path, direct_pages)
                return ConversionResult(
                    markdown="",
                    pages=[],
                    source_file=str(file_path),
                    total_pages=0,
                )

            ocr_results = self.pipeline.process_pages_sync(pages)
            return self._build_result(
                file_path, ocr_results, direct_pages, len(pages)
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    async def convert_async(
        self,
        file_path: str | Path,
        *,
        model_client: ModelClient | None = None,
        parse_mode: str | None = None,
    ) -> ConversionResult:
        """异步转换接口"""
        result: ConversionResult | None = None
        async for event in self.convert_stream(
            file_path, model_client=model_client, parse_mode=parse_mode
        ):
            if event["type"] == "done":
                result = event["result"]
        assert result is not None
        return result

    @staticmethod
    def _result_events(result: ConversionResult) -> list[dict]:
        """Events for a result that is complete up front (no OCR to wait on)."""
        return [
            {"type": "meta", "total_pages": result.total_pages, "ocr_pages": []},
            *({"type": "page", "page": p} for p in result.pages),
            {"type": "done", "result": result},
        ]

    async def convert_stream(
        self,
        file_path: str | Path,
        *,
        model_client: ModelClient | None = None,
        parse_mode: str | None = None,
    ) -> AsyncIterator[dict]:
        """异步转换接口，按页面完成顺序产出进度事件

        ``model_client`` / ``parse_mode`` override the instance defaults for
        this call only.

        事件（以 ``type`` 区分）：
            {"type": "stage", "stage": "parsing" | "rendering" | "ocr"}
            {"type": "meta", "total_pages": int, "ocr_pages": list[int]}
            {"type": "page", "page": PageResult}
            {"type": "done", "result": ConversionResult}
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = detect_file_type(file_path)
        renderer = self._renderers.get(file_type)
        if renderer is None:
            raise UnsupportedFileTypeError(f"No renderer for type: {file_type}")

        parse_mode = parse_mode or self.parse_mode
        if parse_mode not in PARSE_MODES:
            raise ValueError(
                f"Invalid parse_mode: {parse_mode!r}; expected one of {PARSE_MODES}"
            )

        # Per-call pipeline: requests may use different models, and the debug
        # dir must not leak between concurrent conversions.
        pipeline = Pipeline(
            model_client=model_client or self.model_client,
            concurrency=self.pipeline.concurrency,
            max_retries=self.pipeline.max_retries,
            debug_dir=self._get_file_debug_dir(file_path),
            table_format=self.pipeline.table_format,
        )

        if parse_mode != "ocr":
            yield {"type": "stage", "stage": "parsing"}
        direct_pages = await self._direct_pages_async(file_path, file_type, parse_mode)
        if direct_pages is not None and all(
            p.markdown is not None for p in direct_pages
        ):
            for event in self._result_events(
                self._direct_result(file_path, direct_pages)
            ):
                yield event
            return

        yield {"type": "stage", "stage": "rendering"}
        tmp_dir = create_temp_dir()
        try:
            render_kwargs: dict = {"dpi": self.dpi}
            if direct_pages is not None:
                render_kwargs["page_indices"] = [
                    p.page_index for p in direct_pages if p.markdown is None
                ]

            pages: list[PageImage] = await asyncio.to_thread(
                renderer.render, file_path, tmp_dir, **render_kwargs
            )

            if not pages:
                if direct_pages is not None:
                    result = self._direct_result(file_path, direct_pages)
                else:
                    result = ConversionResult(
                        markdown="",
                        pages=[],
                        source_file=str(file_path),
                        total_pages=0,
                    )
                for event in self._result_events(result):
                    yield event
                return

            yield {
                "type": "meta",
                "total_pages": (
                    len(direct_pages) if direct_pages is not None else len(pages)
                ),
                "ocr_pages": [p.page_index for p in pages],
            }
            for p in direct_pages or []:
                if p.markdown is not None:
                    yield {
                        "type": "page",
                        "page": PageResult(page_index=p.page_index, markdown=p.markdown),
                    }

            yield {"type": "stage", "stage": "ocr"}
            ocr_results: list[PageResult] = []
            async for r in pipeline.iter_pages(pages):
                ocr_results.append(r)
                yield {"type": "page", "page": r}

            yield {
                "type": "done",
                "result": self._build_result(
                    file_path, ocr_results, direct_pages, len(pages)
                ),
            }
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
