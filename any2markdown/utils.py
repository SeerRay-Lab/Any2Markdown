import shutil
import tempfile
from pathlib import Path

from any2markdown.models import FileType


EXTENSION_MAP: dict[str, FileType] = {
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".png": FileType.IMAGE,
    ".webp": FileType.IMAGE,
    ".tiff": FileType.IMAGE,
    ".tif": FileType.IMAGE,
    ".bmp": FileType.IMAGE,
    ".pdf": FileType.PDF,
    ".doc": FileType.DOC,
    ".docx": FileType.DOCX,
    ".ppt": FileType.PPT,
    ".pptx": FileType.PPTX,
    ".xlsx": FileType.XLSX,
}

MAGIC_SIGNATURES: list[tuple[bytes, FileType]] = [
    (b"%PDF", FileType.PDF),
    (b"\x89PNG\r\n\x1a\n", FileType.IMAGE),
    (b"\xff\xd8\xff", FileType.IMAGE),
    (b"GIF87a", FileType.IMAGE),
    (b"GIF89a", FileType.IMAGE),
    (b"BM", FileType.IMAGE),
    (b"RIFF", FileType.IMAGE),  # WEBP starts with RIFF
    (b"II\x2a\x00", FileType.IMAGE),  # TIFF little-endian
    (b"MM\x00\x2a", FileType.IMAGE),  # TIFF big-endian
    # ZIP-based formats (DOCX, PPTX, XLSX all start with PK)
    (b"PK\x03\x04", None),  # needs further inspection
]


class UnsupportedFileTypeError(Exception):
    pass


class RendererNotAvailableError(Exception):
    pass


class RenderError(Exception):
    pass


class DirectParseError(Exception):
    """Raised when direct parsing is required but the file has no usable text."""


def detect_file_type(file_path: Path) -> FileType:
    """检测文件类型，优先用扩展名，fallback 用 magic bytes"""
    ext = file_path.suffix.lower()
    if ext in EXTENSION_MAP:
        return EXTENSION_MAP[ext]

    with open(file_path, "rb") as f:
        header = f.read(16)

    for magic, file_type in MAGIC_SIGNATURES:
        if header.startswith(magic):
            if file_type is not None:
                return file_type
            # PK header: ZIP-based, need extension to distinguish
            if ext in (".docx", ".pptx", ".xlsx"):
                return EXTENSION_MAP[ext]
            break

    raise UnsupportedFileTypeError(
        f"Unsupported file type: {file_path.name}. "
        f"Supported formats: {', '.join(sorted(EXTENSION_MAP.keys()))}"
    )


def check_libreoffice() -> str | None:
    """检查 LibreOffice 是否可用，返回可执行文件路径"""
    for cmd in ("libreoffice", "soffice"):
        path = shutil.which(cmd)
        if path:
            return path
    return None


def create_temp_dir(prefix: str = "any2md_") -> Path:
    """创建临时目录"""
    return Path(tempfile.mkdtemp(prefix=prefix))
