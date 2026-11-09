from abc import ABC, abstractmethod
from pathlib import Path

from any2markdown.models import PageImage


class BaseRenderer(ABC):
    @abstractmethod
    def render(self, file_path: Path, output_dir: Path, **kwargs) -> list[PageImage]:
        """将文件渲染为图片列表"""
        ...
