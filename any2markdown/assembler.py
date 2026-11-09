from any2markdown.models import PageResult


class Assembler:
    def assemble(self, pages: list[PageResult]) -> str:
        """按页码顺序合并 Markdown，失败页面用注释占位"""
        sorted_pages = sorted(pages, key=lambda p: p.page_index)

        if len(sorted_pages) == 1:
            p = sorted_pages[0]
            if p.success:
                return p.markdown
            return f"<!-- Page 1: conversion failed - {p.error} -->"

        parts = []
        for page in sorted_pages:
            if page.success:
                parts.append(page.markdown)
            else:
                parts.append(
                    f"<!-- Page {page.page_index + 1}: conversion failed - {page.error} -->"
                )

        return "\n\n".join(parts)
