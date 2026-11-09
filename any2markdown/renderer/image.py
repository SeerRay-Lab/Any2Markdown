import shutil
from pathlib import Path

from PIL import Image

from any2markdown.models import PageImage
from any2markdown.renderer.base import BaseRenderer


class ImageRenderer(BaseRenderer):
    """图片直通 renderer，处理多页 TIFF 和大图缩放"""

    def __init__(self, max_size: int = 4096):
        self.max_size = max_size

    def render(self, file_path: Path, output_dir: Path, **kwargs) -> list[PageImage]:
        max_size = kwargs.get("max_size", self.max_size)

        with Image.open(file_path) as img:
            n_frames = getattr(img, "n_frames", 1)

            if n_frames == 1:
                out_path = self._process_single(img, file_path, output_dir, max_size)
                return [
                    PageImage(
                        image_path=out_path,
                        page_index=0,
                        total_pages=1,
                        source_file=str(file_path),
                        metadata={"width": img.width, "height": img.height},
                    )
                ]

            # Multi-frame (e.g. multi-page TIFF)
            pages = []
            for i in range(n_frames):
                img.seek(i)
                frame = img.copy()
                out_path = output_dir / f"frame_{i:04d}.png"
                self._save_resized(frame, out_path, max_size)
                pages.append(
                    PageImage(
                        image_path=out_path,
                        page_index=i,
                        total_pages=n_frames,
                        source_file=str(file_path),
                        metadata={"width": frame.width, "height": frame.height},
                    )
                )
            return pages

    def _process_single(
        self, img: Image.Image, file_path: Path, output_dir: Path, max_size: int
    ) -> Path:
        w, h = img.size
        if max(w, h) <= max_size and file_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            # No processing needed, use original file
            out_path = output_dir / file_path.name
            shutil.copy2(file_path, out_path)
            return out_path

        out_path = output_dir / f"{file_path.stem}.png"
        self._save_resized(img, out_path, max_size)
        return out_path

    def _save_resized(self, img: Image.Image, out_path: Path, max_size: int) -> None:
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(out_path, "PNG")
