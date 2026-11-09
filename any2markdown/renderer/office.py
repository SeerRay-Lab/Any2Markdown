import subprocess
import tempfile
import uuid
from pathlib import Path

from any2markdown.models import PageImage
from any2markdown.renderer.base import BaseRenderer
from any2markdown.renderer.pdf import PdfRenderer
from any2markdown.utils import RenderError, RendererNotAvailableError, check_libreoffice


class OfficeRenderer(BaseRenderer):
    """DOC/DOCX/PPT/PPTX/XLSX → PDF (LibreOffice) → page images (PyMuPDF)"""

    def __init__(self, dpi: int = 200, timeout: int = 120):
        self.dpi = dpi
        self.timeout = timeout
        self.pdf_renderer = PdfRenderer(dpi=dpi)

    def render(self, file_path: Path, output_dir: Path, **kwargs) -> list[PageImage]:
        dpi = kwargs.get("dpi", self.dpi)
        self.pdf_renderer.dpi = dpi

        lo_path = check_libreoffice()
        if not lo_path:
            raise RendererNotAvailableError(
                "LibreOffice is required for Office file conversion but was not found.\n"
                "Install it:\n"
                "  Ubuntu/Debian: sudo apt install libreoffice\n"
                "  macOS: brew install --cask libreoffice\n"
                "  Windows: https://www.libreoffice.org/download/"
            )

        pdf_path = self._convert_to_pdf(file_path, lo_path)
        try:
            return self.pdf_renderer.render(pdf_path, output_dir, dpi=dpi)
        finally:
            pdf_path.unlink(missing_ok=True)

    def _convert_to_pdf(self, file_path: Path, lo_path: str) -> Path:
        # Use unique UserInstallation to allow parallel LibreOffice instances
        user_install = Path(tempfile.gettempdir()) / f"any2md_lo_{uuid.uuid4().hex[:8]}"
        pdf_dir = Path(tempfile.mkdtemp(prefix="any2md_pdf_"))

        cmd = [
            lo_path,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(pdf_dir),
            f"-env:UserInstallation=file://{user_install}",
            str(file_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise RenderError(
                f"LibreOffice conversion timed out after {self.timeout}s: {file_path.name}"
            )
        except FileNotFoundError:
            raise RendererNotAvailableError(f"LibreOffice not found at: {lo_path}")
        finally:
            import shutil
            shutil.rmtree(user_install, ignore_errors=True)

        if result.returncode != 0:
            raise RenderError(
                f"LibreOffice conversion failed for {file_path.name}: {result.stderr}"
            )

        pdf_files = list(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            raise RenderError(
                f"LibreOffice produced no PDF output for {file_path.name}"
            )

        return pdf_files[0]
