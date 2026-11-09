from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class FileType(Enum):
    IMAGE = "image"
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    PPT = "ppt"
    PPTX = "pptx"
    XLSX = "xlsx"


@dataclass
class PageImage:
    """模型消费的最小单元：一张图片"""

    image_path: Path
    page_index: int
    total_pages: int
    source_file: str
    metadata: dict = field(default_factory=dict)


@dataclass
class PageResult:
    """单页模型输出"""

    page_index: int
    markdown: str
    success: bool = True
    error: str | None = None


@dataclass
class ConversionResult:
    """最终转换结果"""

    markdown: str
    pages: list[PageResult]
    source_file: str
    total_pages: int
    failed_pages: list[int] = field(default_factory=list)
    # "direct" (parsed without OCR), "ocr", or "hybrid" (both paths used).
    parse_mode_used: str = "ocr"
