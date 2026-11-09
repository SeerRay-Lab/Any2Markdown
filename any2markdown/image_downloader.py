"""Async image downloader for /convert/url localization.

Given a markdown body and the originating URL, find every ``![alt](url)`` and
``<img src="...">`` reference, download the bytes concurrently into a content-
addressed cache directory, and return a localized copy of the markdown plus a
manifest describing each download attempt.

Failures degrade gracefully: the original external URL is preserved in the
localized markdown, and the manifest entry carries a ``status`` describing why.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

from any2markdown.url_fetcher import CT_TO_EXT, _parse_ct

logger = logging.getLogger(__name__)


# Markdown image: ![alt](url) or ![alt](url "title")
_MD_IMG_RE = re.compile(
    r'(!\[[^\]]*\]\()'      # group 1: ![alt](
    r'(\S+?)'                # group 2: url (no whitespace)
    r'((?:\s+["\'][^"\']*["\'])?\s*\))'  # group 3: optional title + )
)
# HTML <img src="..." > / <img src='...' > / <img src=...>
_HTML_IMG_RE = re.compile(
    r'(<img\b[^>]*?\bsrc=)("([^"]*)"|\'([^\']*)\'|([^\s>]+))',
    re.IGNORECASE,
)

# Per-CDN Referer overrides. Hotlink-protected image hosts whitelist a specific
# origin; using the article's source_url often fails when the article was
# reached via a short-link redirector (e.g. xhslink.com → xhscdn 403). Map by
# image host's parent domain — covers all subdomains.
IMAGE_REFERER_BY_HOST: dict[str, str] = {
    "xhscdn.com": "https://www.xiaohongshu.com/",
    "mmbiz.qpic.cn": "https://mp.weixin.qq.com/",
    "zhimg.com": "https://www.zhihu.com/",
    "hdslb.com": "https://www.bilibili.com/",
}


def _referer_for(image_url: str, default: str | None) -> str | None:
    host = (urlparse(image_url).hostname or "").lower()
    for suffix, ref in IMAGE_REFERER_BY_HOST.items():
        if host == suffix or host.endswith("." + suffix):
            return ref
    return default


def _ext_from_url(url: str) -> str:
    path = urlparse(url).path or ""
    return PurePosixPath(path).suffix.lower() if path else ""


def _resolve(base: str, url: str) -> str | None:
    """Resolve a possibly-relative URL against ``base``. Skip data:/javascript:/etc."""
    url = url.strip()
    if not url or url.startswith(("data:", "javascript:", "mailto:", "#")):
        return None
    parsed = urlparse(url)
    if not parsed.scheme:
        return urljoin(base, url)
    if parsed.scheme in ("http", "https"):
        return url
    return None


def _collect_image_urls(markdown: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _MD_IMG_RE.finditer(markdown):
        u = m.group(2).strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    for m in _HTML_IMG_RE.finditer(markdown):
        u = (m.group(3) or m.group(4) or m.group(5) or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


async def _download_one(
    client: httpx.AsyncClient,
    url: str,
    *,
    referer: str | None,
    max_bytes: int,
) -> tuple[bytes, str]:
    headers: dict[str, str] = {}
    if referer:
        headers["Referer"] = referer
    async with client.stream("GET", url, headers=headers) as resp:
        resp.raise_for_status()
        ct = _parse_ct(resp.headers.get("content-type"))
        chunks: list[bytes] = []
        total = 0
        async for chunk in resp.aiter_bytes():
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"image exceeds {max_bytes} bytes")
            chunks.append(chunk)
        return b"".join(chunks), ct


async def download_images(
    markdown: str,
    *,
    source_url: str,
    media_dir: Path,
    public_base_url: str,
    max_image_bytes: int = 5 * 1024 * 1024,
    concurrency: int = 8,
    timeout: float = 10.0,
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
) -> tuple[str, list[dict[str, Any]]]:
    """Download every image referenced in ``markdown`` into ``media_dir``.

    Returns:
        ``(localized_markdown, manifest)`` where ``localized_markdown`` has each
        successfully-downloaded image's URL rewritten to its absolute
        ``public_base_url + /media/<sha256>.<ext>`` form. Failed images keep
        their original URLs.

    Manifest entries always contain ``original_url`` and ``status``; successful
    entries additionally carry ``local_url``, ``sha256``, ``content_type``,
    ``size_bytes``.
    """
    media_dir.mkdir(parents=True, exist_ok=True)
    public_base_url = public_base_url.rstrip("/")

    raw_urls = _collect_image_urls(markdown)
    if not raw_urls:
        return markdown, []

    # Build per-url plan: resolved absolute URL (or None if we should skip).
    plans: dict[str, dict[str, Any]] = {}
    for u in raw_urls:
        resolved = _resolve(source_url, u)
        if resolved is None:
            plans[u] = {"original_url": u, "status": "skipped"}
        else:
            plans[u] = {
                "original_url": u,
                "resolved_url": resolved,
                "status": "pending",
            }

    sem = asyncio.Semaphore(concurrency)
    timeout_obj = httpx.Timeout(timeout, connect=min(timeout, 5.0))

    async with httpx.AsyncClient(
        timeout=timeout_obj,
        follow_redirects=True,
        headers={"User-Agent": user_agent},
    ) as client:

        async def task(orig: str) -> None:
            entry = plans[orig]
            if entry["status"] != "pending":
                return
            resolved: str = entry["resolved_url"]
            async with sem:
                try:
                    content, ct = await _download_one(
                        client, resolved,
                        referer=_referer_for(resolved, source_url),
                        max_bytes=max_image_bytes,
                    )
                except ValueError as e:
                    entry["status"] = "too_large"
                    entry["error"] = str(e)
                    return
                except httpx.HTTPStatusError as e:
                    entry["status"] = f"http_{e.response.status_code}"
                    entry["error"] = str(e)
                    return
                except httpx.HTTPError as e:
                    entry["status"] = "http_error"
                    entry["error"] = str(e)
                    return
                except Exception as e:
                    logger.exception("image download failed: %s", resolved)
                    entry["status"] = "error"
                    entry["error"] = str(e)
                    return

                if ct and not (ct.startswith("image/") or ct == "application/octet-stream"):
                    entry["status"] = "bad_content_type"
                    entry["content_type"] = ct
                    entry["size_bytes"] = len(content)
                    return

                sha = hashlib.sha256(content).hexdigest()
                ext = CT_TO_EXT.get(ct, "") or _ext_from_url(resolved) or ".bin"
                fname = f"{sha}{ext}"
                fpath = media_dir / fname
                if not fpath.exists():
                    try:
                        fpath.write_bytes(content)
                    except OSError as e:
                        entry["status"] = "write_error"
                        entry["error"] = str(e)
                        return

                entry["status"] = "ok"
                entry["sha256"] = sha
                entry["content_type"] = ct
                entry["size_bytes"] = len(content)
                entry["local_url"] = f"{public_base_url}/media/{fname}"

        await asyncio.gather(*(task(u) for u in raw_urls))

    # Substitute URLs in the markdown.
    def replace_md(m: re.Match[str]) -> str:
        u = m.group(2).strip()
        local = plans.get(u, {}).get("local_url")
        if local:
            return f"{m.group(1)}{local}{m.group(3)}"
        return m.group(0)

    def replace_html(m: re.Match[str]) -> str:
        u = (m.group(3) or m.group(4) or m.group(5) or "").strip()
        local = plans.get(u, {}).get("local_url")
        if local:
            return f'{m.group(1)}"{local}"'
        return m.group(0)

    localized = _MD_IMG_RE.sub(replace_md, markdown)
    localized = _HTML_IMG_RE.sub(replace_html, localized)

    # Strip internal helper key from manifest before returning.
    manifest: list[dict[str, Any]] = []
    for u in raw_urls:
        entry = dict(plans[u])
        entry.pop("resolved_url", None)
        manifest.append(entry)
    return localized, manifest
