import asyncio
import json
import logging
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from any2markdown.model_client import ModelClient
from any2markdown.models import PageImage, PageResult
from any2markdown.otsl import render_tables

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        model_client: ModelClient,
        concurrency: int = 4,
        max_retries: int = 3,
        debug_dir: Path | None = None,
        table_format: str = "html",
    ):
        self.model_client = model_client
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.debug_dir = debug_dir
        self.table_format = table_format

    def _save_debug(self, page: PageImage, markdown: str, error: str | None = None) -> None:
        if self.debug_dir is None:
            return
        page_dir = self.debug_dir / f"page_{page.page_index:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        dest_img = page_dir / page.image_path.name
        if not dest_img.exists():
            shutil.copy2(page.image_path, dest_img)

        (page_dir / "prompt.txt").write_text(self.model_client.prompt, encoding="utf-8")
        (page_dir / "response.md").write_text(markdown, encoding="utf-8")

        meta = {
            "page_index": page.page_index,
            "total_pages": page.total_pages,
            "source_file": page.source_file,
            "image_file": str(dest_img.name),
            "model": self.model_client.model,
            "error": error,
            "metadata": page.metadata,
        }
        (page_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def process_pages(self, pages: list[PageImage]) -> list[PageResult]:
        """并发处理所有页面"""
        semaphore = asyncio.Semaphore(self.concurrency)

        async def process_one(page: PageImage) -> PageResult:
            async with semaphore:
                return await self._process_with_retry(page)

        tasks = [process_one(page) for page in pages]
        return await asyncio.gather(*tasks)

    async def iter_pages(self, pages: list[PageImage]) -> AsyncIterator[PageResult]:
        """并发处理页面，按完成顺序逐个产出结果

        消费方提前退出（例如客户端断开）时，取消尚未完成的页面。
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def process_one(page: PageImage) -> PageResult:
            async with semaphore:
                return await self._process_with_retry(page)

        tasks = [asyncio.ensure_future(process_one(page)) for page in pages]
        try:
            for fut in asyncio.as_completed(tasks):
                yield await fut
        finally:
            for t in tasks:
                t.cancel()

    async def _process_with_retry(self, page: PageImage) -> PageResult:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                markdown = await self.model_client.image_to_markdown_async(
                    page.image_path
                )
                markdown = render_tables(markdown, self.table_format)
                self._save_debug(page, markdown)
                return PageResult(
                    page_index=page.page_index,
                    markdown=markdown,
                    success=True,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(
                        f"Page {page.page_index} attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)

        logger.error(f"Page {page.page_index} failed after {self.max_retries} attempts")
        self._save_debug(page, "", error=str(last_error))
        return PageResult(
            page_index=page.page_index,
            markdown="",
            success=False,
            error=str(last_error),
        )

    def process_pages_sync(self, pages: list[PageImage]) -> list[PageResult]:
        """同步入口"""
        return asyncio.run(self.process_pages(pages))
