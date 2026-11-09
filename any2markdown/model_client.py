import base64
import mimetypes
from copy import deepcopy
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI


DEFAULT_PROMPT = """Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in OTSL format, formulas should be represented using LATEX format, and the parsing should be organized according to the reading order."""

# when "thinking" is on for OCR-style tasks; turn it off by default.
DEFAULT_EXTRA_BODY: dict[str, Any] = {
    "thinking": {"type": "disabled"},
}


class ModelClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "htt://localhost:8000/v1",
        model: str = "xiaomi-ocr-0",
        prompt: str | None = None,
        temperature: float | None = None,
        timeout: int = 60,
        extra_body: dict[str, Any] | None = None,
    ):
        self.model = model
        self.prompt = prompt or DEFAULT_PROMPT
        self.temperature = temperature
        self.timeout = timeout
        self.extra_body: dict[str, Any] = (
            deepcopy(extra_body) if extra_body is not None else deepcopy(DEFAULT_EXTRA_BODY)
        )
        self._sync_client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
        self._async_client = AsyncOpenAI(api_key=api_key or "EMPTY", base_url=base_url)

    def _build_kwargs(self) -> dict:
        kwargs: dict = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.extra_body:
            kwargs["extra_body"] = deepcopy(self.extra_body)
        return kwargs

    def image_to_markdown(self, image_path: Path) -> str:
        data_url = self._encode_image(image_path)
        response = self._sync_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            timeout=self.timeout,
            **self._build_kwargs(),
        )
        return response.choices[0].message.content or ""

    async def image_to_markdown_async(self, image_path: Path) -> str:
        data_url = self._encode_image(image_path)
        response = await self._async_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            timeout=self.timeout,
            **self._build_kwargs(),
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        mime_type = mimetypes.guess_type(str(image_path))[0] or "image/png"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"
