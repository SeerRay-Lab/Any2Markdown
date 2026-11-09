from pathlib import Path

try:
    import pymupdf as fitz
except ImportError:  # older releases only expose fitz
    import fitz

from any2markdown.models import PageImage
from any2markdown.renderer.base import BaseRenderer


class PdfRenderer(BaseRenderer):
    """PDF → page images using PyMuPDF"""

    def __init__(self, dpi: int = 200):
        self.dpi = dpi

    def render(self, file_path: Path, output_dir: Path, **kwargs) -> list[PageImage]:
        dpi = kwargs.get("dpi", self.dpi)
        page_indices = kwargs.get("page_indices")
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        doc = fitz.open(file_path)
        total_pages = len(doc)
        pages = []

        indices = (
            range(total_pages)
            if page_indices is None
            else sorted(i for i in page_indices if 0 <= i < total_pages)
        )

        for i in indices:
            page = doc[i]
            pix = page.get_pixmap(matrix=matrix)
            out_path = output_dir / f"page_{i:04d}.png"
            pix.save(str(out_path))
            pages.append(
                PageImage(
                    image_path=out_path,
                    page_index=i,
                    total_pages=total_pages,
                    source_file=str(file_path),
                    metadata={
                        "dpi": dpi,
                        "width": pix.width,
                        "height": pix.height,
                    },
                )
            )

        doc.close()
        return pages
