"""FastAPI HTTP service for Any2Markdown.

Endpoints:
    GET  /health              → service + model info
    GET  /models              → model profiles (API keys masked)
    GET  /models/{id}/status  → probe the profile's API ({base_url}/models)
    POST /models              → create a model profile
    PUT  /models/{id}         → update a model profile
    DELETE /models/{id}       → delete a model profile
    POST /convert/file        → multipart upload, file → markdown
    POST /convert/file/stream → same as /convert/file, as Server-Sent Events
    POST /convert/url         → JSON {url} → markdown
    POST /convert/url/stream  → same as /convert/url, as Server-Sent Events
    POST /convert/text        → JSON {text, mime?} → markdown (passthrough/normalize)

The /convert/file and /convert/url endpoints accept two optional per-request
options (form fields for file uploads, JSON fields for URLs):
    model       — model profile id from GET /models (default: the default profile)
    parse_mode  — auto / ocr / direct (default: parse.mode in any2md.yaml)

The ``/stream`` variants emit these SSE events:
    stage  {"stage": "parsing" | "rendering" | "ocr", "elapsed_ms": int}
    meta   {"total_pages": int, "ocr_pages": [int, ...]}
    page   {"page_index": int, "markdown": str, "success": bool, "error": str | null}
    done   same JSON body as the non-streaming endpoint, plus
           "timings": {"fetch_ms"?, "parsing_ms"?, "rendering_ms"?, "ocr_ms"?, "total_ms"}
    error  {"error": str, "status": int}
Pages arrive in completion order, not page order. Errors detected before
conversion starts (bad upload, fetch failure, …) are still returned as plain
HTTP errors with an ``{"error": ...}`` body.

Configuration comes from ``any2md.yaml`` (see :func:`any2markdown.config.load_config`
for the search order; ``ANY2MD_CONFIG`` points at a specific file). Every key
can be overridden by an environment variable named ``ANY2MD_<KEY>``, e.g.
``ANY2MD_BASE_URL``, ``ANY2MD_API_KEY``, ``ANY2MD_PDF_WORKERS``; ``model.name`` is
``ANY2MD_MODEL`` and ``parse.mode`` is ``ANY2MD_PARSE_MODE``.

The model configured there seeds the profile store (``server.profiles_file``,
default data/model_profiles.json) on first start. Once a profile has been
edited in the web UI the file wins, and changing the configured model no
longer affects it.

A single :class:`Any2Markdown` instance is built at process start and reused
across requests. The pipeline already does its own concurrency control so we
don't need to wrap the route handler in another lock.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, NamedTuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from any2markdown.config import load_config
from any2markdown.converter import PARSE_MODES, Any2Markdown
from any2markdown.direct import default_pdf_workers, start_pdf_pool
from any2markdown.headless_fetcher import (
    HeadlessFetchError,
    fetch_html_via_browser,
    pool as browser_pool,
)
from any2markdown.image_downloader import download_images
from any2markdown.model_client import ModelClient
from any2markdown.models import ConversionResult
from any2markdown.profiles import ProfileStore
from any2markdown.url_fetcher import (
    URLFetchError,
    extract_from_html,
    fetch_bytes,
    fetch_url,
    infer_file_suffix,
    is_html_content_type,
    site_needs_headless,
    site_user_agent,
)
from any2markdown.utils import (
    DirectParseError,
    UnsupportedFileTypeError,
    detect_file_type,
)

_cfg = load_config()

logger = logging.getLogger("any2markdown.server")
logging.basicConfig(level=_cfg.server.log_level)


# ---- Settings ----------------------------------------------------------------


class Settings:
    """Flattened view of the loaded config used by the routes below."""

    base_url: str = _cfg.model.base_url
    api_key: str | None = _cfg.model.api_key or None
    model: str = _cfg.model.name
    dpi: int = _cfg.render.dpi
    concurrency: int = _cfg.pipeline.concurrency
    max_retries: int = _cfg.pipeline.max_retries
    timeout: int = _cfg.model.timeout
    prompt: str | None = _cfg.model.prompt or None
    temperature: float = (
        _cfg.model.temperature if _cfg.model.temperature is not None else 0.0
    )
    table_format: str = _cfg.pipeline.table_format
    parse_mode: str = _cfg.parse.mode
    host: str = _cfg.server.host
    port: int = _cfg.server.port
    url_timeout: float = _cfg.server.url_timeout
    max_upload_bytes: int = _cfg.server.max_upload_mb * 1024 * 1024
    pdf_workers: int = (
        _cfg.server.pdf_workers
        if _cfg.server.pdf_workers is not None
        else default_pdf_workers()
    )
    profiles_file: Path = Path(_cfg.server.profiles_file)

    # Image download (only used when /convert/url is called with download_images=true)
    media_dir: Path = Path(_cfg.server.media_dir)
    public_base_url: str | None = _cfg.server.public_base_url or None
    image_max_bytes: int = _cfg.server.image_max_mb * 1024 * 1024
    image_concurrency: int = _cfg.server.image_concurrency
    image_timeout: float = _cfg.server.image_timeout

    # Seconds of silence before a streaming response sends an SSE keep-alive
    stream_heartbeat: float = _cfg.server.stream_heartbeat


# ---- Models ------------------------------------------------------------------


class ConvertURLRequest(BaseModel):
    url: str = Field(..., description="HTTP(S) URL to fetch and convert.")
    download_images: bool = Field(
        False,
        description=(
            "When true, download every image referenced by the article into "
            "the server's media cache and return a `markdown_localized` field "
            "with rewritten absolute URLs alongside the original `markdown`."
        ),
    )
    max_image_bytes: int | None = Field(
        None, description="Per-image size cap in bytes. Defaults to server config."
    )
    image_concurrency: int | None = Field(
        None, description="Max concurrent image downloads. Defaults to server config."
    )
    model: str | None = Field(
        None, description="Model profile id (GET /models). Defaults to the default profile."
    )
    parse_mode: str | None = Field(
        None, description="auto / ocr / direct. Defaults to server config."
    )


class ModelProfileRequest(BaseModel):
    name: str = Field(..., description="Display name.")
    model: str = Field(..., description="Model id sent to the API.")
    base_url: str = Field(..., description="OpenAI-compatible API base URL.")
    api_key: str | None = Field(
        None, description="New API key; omit or null to keep the stored one."
    )


class ConvertTextRequest(BaseModel):
    text: str = Field(..., description="Text content. If MIME is markdown, returned as-is.")
    mime: str | None = Field(
        None, description="Optional MIME type (text/markdown, text/plain, text/html)."
    )


class _FileJob(NamedTuple):
    """A downloaded binary that still has to go through the file pipeline."""

    tmp_dir: Path
    tmp_path: Path
    source: dict[str, Any]


# ---- App factory -------------------------------------------------------------


def build_converter(settings: Settings = Settings) -> Any2Markdown:
    """Build the singleton converter from Settings."""
    return Any2Markdown(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        prompt=settings.prompt,
        temperature=settings.temperature,
        timeout=settings.timeout,
        dpi=settings.dpi,
        concurrency=settings.concurrency,
        max_retries=settings.max_retries,
        table_format=settings.table_format,
        parse_mode=settings.parse_mode,
    )


def build_profile_store(settings: Settings = Settings) -> ProfileStore:
    """Model profiles, seeded from the configured model."""
    return ProfileStore(
        settings.profiles_file,
        seed={
            "id": "default",
            "name": settings.model,
            "model": settings.model,
            "base_url": settings.base_url,
            "api_key": settings.api_key,
        },
        client_factory=lambda p: ModelClient(
            api_key=p.get("api_key"),
            base_url=p["base_url"],
            model=p["model"],
            prompt=settings.prompt,
            temperature=settings.temperature,
            timeout=settings.timeout,
        ),
    )


def create_app(converter: Any2Markdown | None = None) -> FastAPI:
    """Build the FastAPI app. Pass a custom ``converter`` to override the singleton."""
    app = FastAPI(
        title="Any2Markdown HTTP",
        version="0.1.0",
        description="File / URL / text → Markdown service.",
    )

    if converter is None:
        converter = build_converter()
    app.state.converter = converter
    app.state.profiles = build_profile_store()

    @app.on_event("startup")
    def _start_pdf_pool() -> None:
        # Warm the PDF worker processes in the background so start-up isn't
        # delayed and the first request doesn't pay for spawning them.
        threading.Thread(
            target=start_pdf_pool, args=(Settings.pdf_workers,), daemon=True
        ).start()

    # ---- routes --------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": Settings.model,
            "base_url": Settings.base_url,
            "has_api_key": bool(Settings.api_key),
            "table_format": Settings.table_format,
            "parse_mode": Settings.parse_mode,
        }

    @app.get("/models")
    def list_models(request: Request) -> dict[str, Any]:
        return request.app.state.profiles.public()

    @app.get("/models/{profile_id}/status")
    async def model_status(request: Request, profile_id: str) -> dict[str, Any]:
        store: ProfileStore = request.app.state.profiles
        profile = store.get(profile_id)
        if profile is None:
            raise HTTPException(status_code=404, detail={"error": "模型不存在"})
        return await _probe_model(profile)

    @app.post("/models")
    def create_model(request: Request, body: ModelProfileRequest) -> dict[str, Any]:
        return _save_profile(request, None, body)

    @app.put("/models/{profile_id}")
    def update_model(
        request: Request, profile_id: str, body: ModelProfileRequest
    ) -> dict[str, Any]:
        return _save_profile(request, profile_id, body)

    @app.delete("/models/{profile_id}")
    def delete_model(request: Request, profile_id: str) -> dict[str, Any]:
        store: ProfileStore = request.app.state.profiles
        try:
            store.delete(profile_id)
        except KeyError:
            raise HTTPException(status_code=404, detail={"error": "模型不存在"})
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})
        except OSError as e:
            raise HTTPException(status_code=500, detail={"error": f"保存模型配置失败：{e}"})
        return store.public()

    @app.post("/convert/file")
    async def convert_file(
        request: Request,
        file: UploadFile = File(...),
        model: str | None = Form(None),
        parse_mode: str | None = Form(None),
    ) -> JSONResponse:
        options = _conversion_options(request, model, parse_mode)
        tmp_dir, tmp_path = await _save_upload(file)
        try:
            t0 = time.perf_counter()
            try:
                result = await request.app.state.converter.convert_async(
                    tmp_path, **options
                )
            except Exception as e:
                raise _conversion_http_error(e, "convert/file")
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            return JSONResponse(
                {
                    **_result_payload(result),
                    "elapsed_ms": elapsed_ms,
                    "source_filename": file.filename,
                }
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @app.post("/convert/file/stream")
    async def convert_file_stream(
        request: Request,
        file: UploadFile = File(...),
        model: str | None = Form(None),
        parse_mode: str | None = Form(None),
    ) -> StreamingResponse:
        options = _conversion_options(request, model, parse_mode)
        tmp_dir, tmp_path = await _save_upload(file)
        return _sse_response(
            _stream_conversion(
                request.app.state.converter,
                tmp_path,
                options=options,
                extra={"source_filename": file.filename},
                t0=time.perf_counter(),
                where="convert/file/stream",
            ),
            cleanup_dir=tmp_dir,
        )

    @app.post("/convert/url")
    async def convert_url(request: Request, body: ConvertURLRequest) -> JSONResponse:
        options = _conversion_options(request, body.model, body.parse_mode)
        t0 = time.perf_counter()
        target = await _resolve_url(request, body, t0)
        if not isinstance(target, _FileJob):
            return JSONResponse(target)

        try:
            try:
                result = await request.app.state.converter.convert_async(
                    target.tmp_path, **options
                )
            except Exception as e:
                raise _conversion_http_error(e, "convert/url file conversion")
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            return JSONResponse(
                {
                    **_result_payload(result),
                    "source": target.source,
                    "elapsed_ms": elapsed_ms,
                }
            )
        finally:
            shutil.rmtree(target.tmp_dir, ignore_errors=True)

    @app.post("/convert/url/stream")
    async def convert_url_stream(
        request: Request, body: ConvertURLRequest
    ) -> StreamingResponse:
        options = _conversion_options(request, body.model, body.parse_mode)
        t0 = time.perf_counter()
        target = await _resolve_url(request, body, t0)
        fetch_ms = int((time.perf_counter() - t0) * 1000)
        if not isinstance(target, _FileJob):
            # HTML pages are extracted in one go — nothing to stream page by page.
            async def single() -> AsyncIterator[str]:
                yield _sse("done", {**target, "timings": {"fetch_ms": fetch_ms, "total_ms": fetch_ms}})

            return _sse_response(single())

        return _sse_response(
            _stream_conversion(
                request.app.state.converter,
                target.tmp_path,
                options=options,
                extra={"source": target.source},
                t0=t0,
                where="convert/url/stream",
                timings={"fetch_ms": fetch_ms},
            ),
            cleanup_dir=target.tmp_dir,
        )

    @app.post("/convert/text")
    async def convert_text(body: ConvertTextRequest) -> JSONResponse:
        t0 = time.perf_counter()
        text = body.text or ""
        mime = (body.mime or "").lower().strip()

        if mime in ("text/html", "html"):
            try:
                from any2markdown.url_fetcher import _generic_extract

                md, _ = _generic_extract("about:blank", text)
            except Exception as e:
                raise HTTPException(status_code=400, detail={"error": str(e)})
        else:
            # text/markdown or text/plain or unspecified → passthrough,
            # but normalise CRLF and strip BOM.
            md = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return JSONResponse({"markdown": md, "elapsed_ms": elapsed_ms})

    @app.exception_handler(HTTPException)
    async def _http_exc(_request: Request, exc: HTTPException) -> JSONResponse:
        # Always return {"error": "..."} shape
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            payload = detail
        else:
            payload = {"error": str(detail)}
        return JSONResponse(payload, status_code=exc.status_code)

    # Mount /media (image cache) before "/" so it wins prefix matching.
    media_dir = Settings.media_dir
    try:
        media_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/media", StaticFiles(directory=media_dir), name="media")
    except OSError as e:
        logger.warning("media dir %s not writable; /media disabled: %s", media_dir, e)

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


# ---- Route helpers -----------------------------------------------------------


async def _probe_model(profile: dict[str, Any]) -> dict[str, Any]:
    """Check that the profile's OpenAI-compatible API answers ``GET /models``.

    ``model_found`` is ``None`` when the server's model list can't be read.
    """
    import httpx

    headers = {}
    if profile.get("api_key"):
        headers["Authorization"] = f"Bearer {profile['api_key']}"
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{profile['base_url']}/models", headers=headers)
    except httpx.HTTPError as e:
        return {"online": False, "error": f"无法连接（{type(e).__name__}）"}
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if resp.status_code in (401, 403):
        return {"online": False, "latency_ms": latency_ms, "error": f"鉴权失败（HTTP {resp.status_code}），请检查 API Key"}
    if resp.status_code >= 400:
        return {"online": False, "latency_ms": latency_ms, "error": f"HTTP {resp.status_code}"}

    model_found = None
    try:
        ids = [m.get("id") for m in resp.json().get("data", [])]
        if ids:
            model_found = profile["model"] in ids
    except (ValueError, AttributeError):
        pass
    result: dict[str, Any] = {"online": True, "latency_ms": latency_ms, "model_found": model_found}
    if model_found is False:
        result["error"] = f"服务在线，但未提供模型 {profile['model']}"
    return result


def _save_profile(
    request: Request, profile_id: str | None, body: ModelProfileRequest
) -> dict[str, Any]:
    fields = {
        "name": body.name.strip(),
        "model": body.model.strip(),
        "base_url": body.base_url.strip().rstrip("/"),
        "api_key": body.api_key.strip() if body.api_key is not None else None,
    }
    if not fields["name"] or not fields["model"]:
        raise HTTPException(status_code=400, detail={"error": "名称和模型名不能为空"})
    if not fields["base_url"].startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400, detail={"error": "Base URL 需以 http:// 或 https:// 开头"}
        )

    store: ProfileStore = request.app.state.profiles
    try:
        saved_id = store.save(profile_id, fields)
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "模型不存在"})
    except OSError as e:
        raise HTTPException(status_code=500, detail={"error": f"保存模型配置失败：{e}"})
    return {**store.public(), "id": saved_id}


def _conversion_options(
    request: Request, model: str | None, parse_mode: str | None
) -> dict[str, Any]:
    """Validate per-request options into ``convert_*`` keyword arguments."""
    options: dict[str, Any] = {}
    if parse_mode:
        if parse_mode not in PARSE_MODES:
            raise HTTPException(
                status_code=400,
                detail={"error": f"invalid parse_mode {parse_mode!r}; expected one of {PARSE_MODES}"},
            )
        options["parse_mode"] = parse_mode
    if model:
        try:
            options["model_client"] = request.app.state.profiles.client(model)
        except KeyError:
            raise HTTPException(
                status_code=400, detail={"error": f"模型不存在：{model}，请刷新页面后重新选择"}
            )
    return options


def _result_payload(result: ConversionResult) -> dict[str, Any]:
    return {
        "markdown": result.markdown,
        "total_pages": result.total_pages,
        "failed_pages": result.failed_pages,
        "parse_mode_used": result.parse_mode_used,
    }


def _conversion_http_error(e: Exception, where: str) -> HTTPException:
    """Map a converter exception to the HTTP error the endpoints return."""
    if isinstance(e, UnsupportedFileTypeError):
        return HTTPException(status_code=415, detail={"error": str(e)})
    if isinstance(e, DirectParseError):
        return HTTPException(status_code=422, detail={"error": str(e)})
    logger.exception("%s failed", where)
    return HTTPException(status_code=500, detail={"error": str(e)})


async def _save_upload(file: UploadFile) -> tuple[Path, Path]:
    """Stream an upload to a temp dir, capped at max_upload_bytes.

    Returns ``(tmp_dir, tmp_path)``; the caller owns ``tmp_dir``. On error the
    dir is removed and an HTTPException raised.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail={"error": "missing filename"})

    suffix = Path(file.filename).suffix or ""
    tmp_dir = Path(tempfile.mkdtemp(prefix="any2md_upload_"))
    tmp_path = tmp_dir / f"upload{suffix}"
    bytes_seen = 0
    try:
        with tmp_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_seen += len(chunk)
                if bytes_seen > Settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "error": (
                                f"upload exceeds {Settings.max_upload_bytes} bytes"
                            )
                        },
                    )
                out.write(chunk)

        # Up-front file type sanity check so we 4xx instead of 5xx
        try:
            detect_file_type(tmp_path)
        except UnsupportedFileTypeError as e:
            raise HTTPException(status_code=415, detail={"error": str(e)})
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return tmp_dir, tmp_path


async def _resolve_url(
    request: Request, body: ConvertURLRequest, t0: float
) -> dict[str, Any] | _FileJob:
    """Fetch ``body.url``.

    HTML pages are extracted right away and returned as the finished response
    payload. Binary documents are written to a temp file and returned as a
    :class:`_FileJob` for the file pipeline; the caller owns its ``tmp_dir``.
    """
    url = body.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail={"error": "url is empty"})

    import anyio

    # Branch 0: sites with JS challenges (e.g. Zhihu zse-ck) — go straight
    # to a real Chromium so the challenge resolves on its own.
    headless_cfg = site_needs_headless(url)
    if headless_cfg:
        try:
            browser = await browser_pool.get()
            html = await fetch_html_via_browser(
                browser,
                url,
                timeout=max(Settings.url_timeout, 30.0),
                wait_for_selector=headless_cfg.get("wait_for"),
            )
        except HeadlessFetchError as e:
            raise HTTPException(status_code=502, detail={"error": str(e)})
        except Exception as e:
            logger.exception("convert/url headless fetch failed")
            raise HTTPException(status_code=500, detail={"error": str(e)})

        try:
            md, meta = await anyio.to_thread.run_sync(
                lambda: extract_from_html(url, html)
            )
        except Exception as e:
            logger.exception("convert/url headless extract failed")
            raise HTTPException(status_code=500, detail={"error": str(e)})

        # Auth-wall: page rendered but content is gated behind login.
        if meta.get("auth_required") or (
            not (meta.get("title") or md.strip())
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": (
                        "无法提取该链接的正文：页面要求登录才能查看完整内容。"
                        "请尝试登录后复制原文使用 /convert/text 端点，或提供"
                        "公开可访问的链接。"
                    ),
                    "kind": "auth_required",
                    "site_type": meta.get("site_type"),
                },
            )

        payload: dict[str, Any] = {
            "markdown": md,
            "source": {
                "url": meta.get("source_url", url),
                "title": meta.get("title"),
                "author": meta.get("author"),
                "publish_time": meta.get("publish_time"),
                "site_type": meta.get("site_type"),
                "captured_at": meta.get("captured_at"),
                "kind": "html",
                "fetcher": "headless",
                "truncated": bool(meta.get("truncated")),
            },
        }
        await _maybe_download_images(request, body, payload, md, meta.get("source_url", url))
        payload["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        return payload

    # Stream the response. Cap at max_upload_bytes (same limit as
    # /convert/file) so large PDFs (e.g. arxiv papers) fit.
    try:
        fetched = await anyio.to_thread.run_sync(
            lambda: fetch_bytes(
                url,
                max_bytes=Settings.max_upload_bytes,
                timeout=max(Settings.url_timeout, 60.0),
                user_agent=site_user_agent(url),
            )
        )
    except URLFetchError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})
    except Exception as e:
        logger.exception("convert/url fetch failed")
        raise HTTPException(status_code=500, detail={"error": str(e)})

    ct = fetched["content_type"]
    raw: bytes = fetched["content"]
    encoding = fetched.get("encoding") or "utf-8"

    # Branch 1: HTML → existing extractor pipeline.
    if is_html_content_type(ct) or (not ct and raw[:1] in (b"<", b" ", b"\n", b"\r")):
        try:
            html = raw.decode(encoding, errors="replace")
        except LookupError:
            html = raw.decode("utf-8", errors="replace")
        try:
            md, meta = await anyio.to_thread.run_sync(
                lambda: extract_from_html(url, html)
            )
        except URLFetchError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})
        except Exception as e:
            logger.exception("convert/url html extract failed")
            raise HTTPException(status_code=500, detail={"error": str(e)})

        payload = {
            "markdown": md,
            "source": {
                "url": meta.get("source_url", url),
                "title": meta.get("title"),
                "author": meta.get("author"),
                "publish_time": meta.get("publish_time"),
                "site_type": meta.get("site_type"),
                "captured_at": meta.get("captured_at"),
                "kind": "html",
            },
        }
        await _maybe_download_images(request, body, payload, md, meta.get("source_url", url))
        payload["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        return payload

    # Branch 2: binary file → write to temp, hand to the file pipeline.
    suffix = infer_file_suffix(ct, fetched.get("filename"), url)
    tmp_dir = Path(tempfile.mkdtemp(prefix="any2md_url_"))
    tmp_path = tmp_dir / f"download{suffix}"
    try:
        tmp_path.write_bytes(raw)
        try:
            detect_file_type(tmp_path)
        except UnsupportedFileTypeError as e:
            raise HTTPException(
                status_code=415,
                detail={
                    "error": (
                        f"Unsupported content from URL "
                        f"(content-type={ct!r}, suffix={suffix!r}): {e}"
                    )
                },
            )
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return _FileJob(
        tmp_dir=tmp_dir,
        tmp_path=tmp_path,
        source={
            "url": fetched.get("final_url", url),
            "filename": fetched.get("filename") or tmp_path.name,
            "content_type": ct,
            "size_bytes": len(raw),
            "kind": "file",
        },
    )


async def _maybe_download_images(
    request: Request,
    body: ConvertURLRequest,
    payload: dict[str, Any],
    md: str,
    source_url: str,
) -> None:
    """Add ``markdown_localized``/``images`` (or ``images_error``) to ``payload``."""
    if not body.download_images:
        return
    try:
        localized, manifest = await download_images(
            md,
            source_url=source_url,
            media_dir=Settings.media_dir,
            public_base_url=_public_base_url(request),
            max_image_bytes=body.max_image_bytes or Settings.image_max_bytes,
            concurrency=body.image_concurrency or Settings.image_concurrency,
            timeout=Settings.image_timeout,
        )
        payload["markdown_localized"] = localized
        payload["images"] = manifest
    except Exception as e:
        logger.exception("image download failed")
        payload["images_error"] = str(e)


# ---- Server-Sent Events ------------------------------------------------------


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_response(
    gen: AsyncIterator[str], cleanup_dir: Path | None = None
) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Stop nginx-style proxies from buffering the whole stream.
            "X-Accel-Buffering": "no",
        },
        # Runs even if the client disconnected before the stream started.
        background=(
            BackgroundTask(shutil.rmtree, cleanup_dir, ignore_errors=True)
            if cleanup_dir is not None
            else None
        ),
    )


async def _with_heartbeat(
    agen: AsyncIterator[dict], interval: float
) -> AsyncIterator[dict | None]:
    """Re-yield ``agen``'s items, yielding ``None`` after each ``interval``
    seconds of silence so the caller can send a keep-alive."""

    async def next_item() -> dict:
        return await agen.__anext__()

    pending = asyncio.ensure_future(next_item())
    try:
        while True:
            done, _ = await asyncio.wait({pending}, timeout=interval)
            if not done:
                yield None
                continue
            try:
                item = pending.result()
            except StopAsyncIteration:
                return
            yield item
            pending = asyncio.ensure_future(next_item())
    finally:
        if not pending.done():
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pending
        await agen.aclose()


async def _stream_conversion(
    converter: Any2Markdown,
    path: Path,
    *,
    options: dict[str, Any],
    extra: dict[str, Any],
    t0: float,
    where: str,
    timings: dict[str, int] | None = None,
) -> AsyncIterator[str]:
    """Run ``converter.convert_stream`` and render its events as SSE.

    Also measures how long each stage took; the totals go out with ``done``.
    """
    timings = dict(timings or {})
    stage: str | None = None
    stage_t = time.perf_counter()

    def close_stage() -> None:
        nonlocal stage_t
        now = time.perf_counter()
        if stage is not None:
            key = f"{stage}_ms"
            timings[key] = timings.get(key, 0) + int((now - stage_t) * 1000)
        stage_t = now

    try:
        async for event in _with_heartbeat(
            converter.convert_stream(path, **options), Settings.stream_heartbeat
        ):
            if event is None:
                yield ": keep-alive\n\n"
                continue
            kind = event["type"]
            if kind == "stage":
                close_stage()
                stage = event["stage"]
                yield _sse(
                    "stage",
                    {
                        "stage": stage,
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    },
                )
            elif kind == "page":
                page = event["page"]
                yield _sse(
                    "page",
                    {
                        "page_index": page.page_index,
                        "markdown": page.markdown,
                        "success": page.success,
                        "error": page.error,
                    },
                )
            elif kind == "done":
                close_stage()
                stage = None
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                timings["total_ms"] = elapsed_ms
                logger.info("%s %s timings: %s", where, path.name, timings)
                yield _sse(
                    "done",
                    {
                        **_result_payload(event["result"]),
                        **extra,
                        "elapsed_ms": elapsed_ms,
                        "timings": timings,
                    },
                )
            else:
                yield _sse(kind, {k: v for k, v in event.items() if k != "type"})
    except Exception as e:
        err = _conversion_http_error(e, where)
        yield _sse("error", {**err.detail, "status": err.status_code})


def _public_base_url(request: Request) -> str:
    """Resolve the absolute base URL the client used to reach us.

    Priority: server.public_base_url (config / env) → X-Forwarded-* (proxy) →
    request scheme + Host header. Used to mint absolute /media/... URLs.
    """
    if Settings.public_base_url:
        return Settings.public_base_url.rstrip("/")
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme or "http"
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or "localhost"
    )
    return f"{proto}://{host}"


# ASGI entry point used by uvicorn / docker
app = create_app() if os.environ.get("ANY2MD_NO_AUTOSTART") != "1" else None


def main() -> None:
    """Run via ``python -m any2markdown.server`` (mainly for local debugging)."""
    import uvicorn

    uvicorn.run(
        "any2markdown.server:app", host=Settings.host, port=Settings.port, reload=False
    )


if __name__ == "__main__":
    main()
