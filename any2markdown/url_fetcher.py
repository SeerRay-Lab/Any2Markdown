"""URL → Markdown fetcher.

Public entry point: ``fetch_url(url) -> (markdown, metadata)``.

Strategy:
  1. Detect site type by host. Site-specific parsers exist for WeChat MP,
     Zhihu, Xiaohongshu, Bilibili. Anything else falls back to trafilatura.
  2. The returned markdown is prepended with a YAML frontmatter block
     (``source_url`` / ``title`` / ``author?`` / ``captured_at`` / ``site_type``).

The fetcher is intentionally hardened for low-trust input:
  * 10 second total timeout
  * 5 MiB response cap
  * desktop Chrome User-Agent
  * No JS execution. Sites that hard-require JS (e.g. parts of Xiaohongshu)
    may degrade to a partial parse — that's a known limitation and we still
    return whatever frontmatter we could glean.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ---- Constants ----------------------------------------------------------------

DEFAULT_TIMEOUT = 10.0
MAX_RESPONSE_BYTES = 5 * 1024 * 1024  # 5 MiB
DEFAULT_FETCH_BYTES_TIMEOUT = 60.0

# Content types we treat as HTML for the URL extractor pipeline.
HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "application/xml", "text/xml")

# Content type → file suffix, used when the URL path / Content-Disposition
# don't carry an extension we recognise.
CT_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}
DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
# WeChat MP serves a "环境异常" challenge page to desktop UAs; a mobile
# Safari + MicroMessenger UA gets the real article HTML.
WECHAT_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 "
    "Mobile/15E148 Safari/604.1 MicroMessenger/8.0.40"
)
_SITE_UA: dict[str, str] = {
    "wechat_mp": WECHAT_UA,
}


class URLFetchError(Exception):
    """Raised when a URL cannot be fetched or parsed at all."""


# ---- HTTP helpers -------------------------------------------------------------


def _build_client(
    *, timeout: float = DEFAULT_TIMEOUT, transport: httpx.BaseTransport | None = None
) -> httpx.Client:
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    return httpx.Client(
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
        transport=transport,
    )


def _http_get(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    user_agent: str | None = None,
) -> str:
    """GET ``url`` and return body text, capped at MAX_RESPONSE_BYTES."""
    own_client = client is None
    c = client or _build_client(timeout=timeout)
    req_headers = {"User-Agent": user_agent} if user_agent else None
    try:
        with c.stream("GET", url, headers=req_headers) as resp:
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise URLFetchError(
                        f"Response exceeded {MAX_RESPONSE_BYTES} bytes for {url}"
                    )
                chunks.append(chunk)
            raw = b"".join(chunks)
            encoding = resp.encoding or "utf-8"
            try:
                return raw.decode(encoding, errors="replace")
            except LookupError:
                return raw.decode("utf-8", errors="replace")
    except httpx.HTTPError as e:
        raise URLFetchError(f"HTTP error fetching {url}: {e}") from e
    finally:
        if own_client:
            c.close()


def _parse_ct(header: str | None) -> str:
    if not header:
        return ""
    return header.split(";", 1)[0].strip().lower()


def _filename_from_disposition(header: str | None) -> str | None:
    if not header:
        return None
    # RFC 5987 filename* takes precedence; fall back to plain filename
    m = re.search(r"filename\*\s*=\s*[^']*''([^;]+)", header, re.I)
    if m:
        from urllib.parse import unquote

        return unquote(m.group(1).strip().strip('"').strip("'"))
    m = re.search(r'filename\s*=\s*"([^"]+)"', header, re.I) or re.search(
        r"filename\s*=\s*([^;]+)", header, re.I
    )
    if m:
        return m.group(1).strip().strip('"').strip("'")
    return None


def is_html_content_type(content_type: str) -> bool:
    return _parse_ct(content_type) in HTML_CONTENT_TYPES


def infer_file_suffix(content_type: str, filename: str | None, url: str) -> str:
    """Pick a file suffix from disposition filename → URL path → content-type."""
    from pathlib import PurePosixPath

    if filename:
        suf = PurePosixPath(filename).suffix.lower()
        if suf:
            return suf
    path = urlparse(url).path or ""
    suf = PurePosixPath(path).suffix.lower()
    if suf:
        return suf
    return CT_TO_EXT.get(_parse_ct(content_type), "")


def fetch_bytes(
    url: str,
    *,
    max_bytes: int,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_FETCH_BYTES_TIMEOUT,
    user_agent: str | None = None,
) -> dict[str, Any]:
    """GET ``url`` with streaming, capped at ``max_bytes``.

    Returns a dict with ``content`` (bytes), ``content_type`` (lower-cased,
    no parameters), ``filename`` (from Content-Disposition, may be None),
    ``encoding`` (response charset hint), ``final_url`` (after redirects).
    """
    own_client = client is None
    c = client or _build_client(timeout=timeout)
    req_headers = {"User-Agent": user_agent} if user_agent else None
    try:
        with c.stream("GET", url, headers=req_headers) as resp:
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in resp.iter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise URLFetchError(
                        f"Response exceeded {max_bytes} bytes for {url}"
                    )
                chunks.append(chunk)
            return {
                "content": b"".join(chunks),
                "content_type": _parse_ct(resp.headers.get("content-type")),
                "filename": _filename_from_disposition(
                    resp.headers.get("content-disposition")
                ),
                "encoding": resp.encoding,
                "final_url": str(resp.url),
            }
    except httpx.HTTPError as e:
        raise URLFetchError(f"HTTP error fetching {url}: {e}") from e
    finally:
        if own_client:
            c.close()


# ---- Site detection -----------------------------------------------------------

_SITE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("wechat_mp", re.compile(r"(^|\.)mp\.weixin\.qq\.com$", re.I)),
    ("zhihu", re.compile(r"(^|\.)zhihu\.com$", re.I)),
    ("xiaohongshu", re.compile(r"(^|\.)(xiaohongshu|xhslink)\.com$", re.I)),
    ("bilibili", re.compile(r"(^|\.)bilibili\.com$", re.I)),
]


def detect_site_type(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    for name, pattern in _SITE_PATTERNS:
        if pattern.search(host):
            return name
    return "generic"


def site_user_agent(url: str) -> str | None:
    """Return a site-specific UA override, or None to use the default."""
    return _SITE_UA.get(detect_site_type(url))


# Per-site config for the headless-browser fallback. A site appearing here
# means: HTTP-only fetching is unreliable for it (JS challenge / heavy SPA
# hydration), so the server should route through Playwright. ``wait_for`` is
# a CSS selector to await before grabbing HTML.
SITE_HEADLESS_CONFIG: dict[str, dict[str, Any]] = {
    "zhihu": {
        # Wait for any of: column-article body, answer body, generic content.
        "wait_for": ".Post-RichText, .RichText, .ContentItem-mainContent, h1.Post-Title",
    },
}


def site_needs_headless(url: str) -> dict[str, Any] | None:
    """Return headless config dict if this URL's site requires JS rendering."""
    return SITE_HEADLESS_CONFIG.get(detect_site_type(url))


# ---- Frontmatter --------------------------------------------------------------


def _yaml_escape(value: str) -> str:
    """Minimal YAML-safe scalar quoting (single-quoted)."""
    return "'" + value.replace("'", "''") + "'"


def _build_frontmatter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for key in ("source_url", "title", "author", "publish_time", "captured_at", "site_type"):
        if key in meta and meta[key]:
            lines.append(f"{key}: {_yaml_escape(str(meta[key]))}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ---- Generic / trafilatura fallback ------------------------------------------


def _generic_extract(url: str, html: str) -> tuple[str, dict[str, Any]]:
    try:
        import trafilatura
    except ImportError as e:
        raise URLFetchError(
            "trafilatura is required for generic URL extraction"
        ) from e

    md = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_recall=True,
    ) or ""

    meta_obj = trafilatura.extract_metadata(html, default_url=url)
    title = ""
    author = ""
    publish_time = ""
    if meta_obj is not None:
        # trafilatura returns a Document object; .as_dict() / attrs differ across versions.
        for attr in ("title",):
            v = getattr(meta_obj, attr, None)
            if v:
                title = str(v)
                break
        v = getattr(meta_obj, "author", None)
        if v:
            author = str(v) if not isinstance(v, list) else ", ".join(v)
        v = getattr(meta_obj, "date", None)
        if v:
            publish_time = str(v)

    metadata: dict[str, Any] = {
        "source_url": url,
        "site_type": "generic",
        "captured_at": _now_iso(),
    }
    if title:
        metadata["title"] = title
    if author:
        metadata["author"] = author
    if publish_time:
        metadata["publish_time"] = publish_time
    return md.strip(), metadata


# ---- Site-specific: WeChat ---------------------------------------------------


def _bs(html: str):
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "html.parser")


def _text(node) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+\n", "\n", node.get_text(separator="\n", strip=True))


def _wechat_extract(url: str, html: str) -> tuple[str, dict[str, Any]]:
    soup = _bs(html)

    title = ""
    h1 = soup.select_one("#activity-name, h1#activity-name, h2.rich_media_title")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()

    author = ""
    a_node = soup.select_one("#js_name, .rich_media_meta_nickname")
    if a_node:
        author = a_node.get_text(strip=True)
    if not author:
        og = soup.find("meta", attrs={"name": "author"})
        if og and og.get("content"):
            author = og["content"].strip()

    publish_time = ""
    p_node = soup.select_one("#publish_time, em#publish_time")
    if p_node:
        publish_time = p_node.get_text(strip=True)
    if not publish_time:
        # fall back to the inline JS variable that WeChat ships
        m = re.search(r'(?:var\s+)?(?:ct|publish_time)\s*=\s*"([^"]+)"', html)
        if m:
            publish_time = m.group(1).strip()

    # Body. WeChat puts the article in #js_content
    body_node = soup.select_one("#js_content, .rich_media_content")
    body_md = ""
    if body_node is not None:
        body_md = _wechat_body_to_md(body_node)

    if not body_md and body_node is not None:
        # Video-only posts (视频号分享): no text body, but the snap card
        # carries data-desc on <mp-common-videosnap>.
        snap = body_node.find(attrs={"data-desc": True})
        if snap and snap.get("data-desc"):
            body_md = f"> [视频] {snap['data-desc'].strip()}"

    if not body_md:
        # 视频号 share posts (appmsg_type=10002) ship no #js_content at all —
        # the text lives in window.cgiDataNew.desc as a JS string literal.
        desc = _wechat_cgi_desc(html)
        if desc:
            body_md = desc

    if not body_md:
        # graceful fallback
        body_md, _ = _generic_extract(url, html)

    metadata: dict[str, Any] = {
        "source_url": url,
        "site_type": "wechat_mp",
        "captured_at": _now_iso(),
    }
    if title:
        metadata["title"] = title
    if author:
        metadata["author"] = author
    if publish_time:
        metadata["publish_time"] = publish_time

    parts = []
    if title:
        parts.append(f"# {title}")
    parts.append(body_md.strip())
    return "\n\n".join(p for p in parts if p), metadata


def _wechat_cgi_desc(html: str) -> str:
    """Extract the article body from ``window.cgiDataNew.desc`` for video-share posts.

    WeChat 视频号 share pages (appmsg_type=10002) carry no #js_content; the
    description text — which IS the article body — is embedded as a JS single-
    quoted string literal inside ``window.cgiDataNew = { ... desc: '...' ... }``.
    """
    # Anchor on cgiDataNew so we don't pick up a desc field from an unrelated
    # script. The desc value is a single-quoted JS string with \xHH escapes
    # (notably \x0a for newlines) and may contain embedded quotes only as \'.
    m = re.search(
        r"window\.cgiDataNew\s*=\s*\{.*?\bdesc:\s*'((?:[^'\\]|\\.)*)'",
        html,
        re.S,
    )
    if not m:
        return ""
    raw = m.group(1)

    def _unesc(match: re.Match[str]) -> str:
        seq = match.group(0)
        if seq.startswith("\\x") and len(seq) == 4:
            try:
                return chr(int(seq[2:], 16))
            except ValueError:
                return seq
        if seq.startswith("\\u") and len(seq) == 6:
            try:
                return chr(int(seq[2:], 16))
            except ValueError:
                return seq
        c = seq[1]
        return {"n": "\n", "r": "\r", "t": "\t", "'": "'", '"': '"', "\\": "\\", "/": "/"}.get(c, c)

    return re.sub(r"\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\.", _unesc, raw).strip()


def _wechat_body_to_md(body_node) -> str:
    out: list[str] = []
    for child in body_node.find_all(
        ["p", "h1", "h2", "h3", "h4", "blockquote", "li", "img"], recursive=True
    ):
        name = child.name
        if name == "img":
            src = child.get("data-src") or child.get("src") or ""
            alt = child.get("alt") or ""
            if src:
                out.append(f"![{alt}]({src})")
            continue
        text = child.get_text(strip=True)
        if not text:
            continue
        if name in ("h1", "h2", "h3", "h4"):
            level = int(name[1])
            out.append("#" * level + " " + text)
        elif name == "blockquote":
            out.append("> " + text.replace("\n", "\n> "))
        elif name == "li":
            out.append("- " + text)
        else:
            out.append(text)
    return "\n\n".join(out)


# ---- Site-specific: Zhihu -----------------------------------------------------


def _zhihu_extract(url: str, html: str) -> tuple[str, dict[str, Any]]:
    soup = _bs(html)
    path = urlparse(url).path

    title = ""
    author = ""
    publish_time = ""
    body_md = ""

    # Two main shapes: /p/<id> column article, /question/<qid>/answer/<aid> answer
    if "/answer/" in path or "/question/" in path:
        answer = soup.select_one("div.AnswerCard, div.AnswerItem, div.RichContent")
        q_title = soup.select_one("h1.QuestionHeader-title")
        if q_title:
            title = q_title.get_text(strip=True)
        author_node = soup.select_one("div.AuthorInfo .UserLink-link, .AuthorInfo-name")
        if author_node:
            author = author_node.get_text(strip=True)
        time_node = soup.select_one("div.ContentItem-time, span[itemprop='dateModified']")
        if time_node:
            publish_time = time_node.get_text(strip=True)
        if answer is not None:
            body_md = _generic_html_to_md(answer)
    else:
        # /p/<id> column article
        title_node = soup.select_one("h1.Post-Title, h1.PostIndex-title")
        if title_node:
            title = title_node.get_text(strip=True)
        author_node = soup.select_one("span.UserLink.AuthorInfo-name a, .AuthorInfo-name")
        if author_node:
            author = author_node.get_text(strip=True)
        time_node = soup.select_one(
            "div.ContentItem-time span, span[itemprop='dateModified']"
        )
        if time_node:
            publish_time = time_node.get_text(strip=True)
        body_node = soup.select_one("div.Post-RichTextContainer, div.RichText")
        if body_node is not None:
            body_md = _generic_html_to_md(body_node)

    if not title:
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()

    if not body_md:
        body_md, _ = _generic_extract(url, html)

    metadata: dict[str, Any] = {
        "source_url": url,
        "site_type": "zhihu",
        "captured_at": _now_iso(),
    }
    if title:
        metadata["title"] = title
    if author:
        metadata["author"] = author
    if publish_time:
        metadata["publish_time"] = publish_time

    # Auth-wall / truncation detection. Zhihu serves three distinct shapes:
    #   ① Public, full DOM rendered → body_md substantial, no truncation marker
    #   ② Truncated: opening + "登录后阅读全文" gate (rest of body is fetched
    #      from a separate API after login)
    #   ③ Hard wall: page is the login interstitial itself; the visible text
    #      is auth-wall copy regardless of length.
    body_clean = body_md.strip()
    truncation_markers = ("登录后阅读全文", "登录后继续阅读", "查看完整文章")
    if any(m in html for m in truncation_markers):
        metadata["truncated"] = True

    # Hard-wall: detect by phrase presence in the EXTRACTED body, not length.
    # Zhihu sometimes returns a couple paragraphs of "请您登录后查看更多..." +
    # boilerplate (>200 chars), so a length threshold is unreliable.
    hard_wall_phrases = (
        "请您登录后查看",
        "登录后查看更多",
        "登录知乎，发现更大的世界",
        "请登录知乎",
    )
    body_is_authwall = (
        not title
        and any(p in body_clean for p in hard_wall_phrases)
    )
    if body_is_authwall:
        metadata["auth_required"] = True
        body_clean = ""  # don't surface auth-wall copy as article content

    parts = []
    if title:
        parts.append(f"# {title}")
    if body_clean:
        parts.append(body_clean)
    return "\n\n".join(parts), metadata


# ---- Site-specific: Xiaohongshu ----------------------------------------------


def _xhs_extract(url: str, html: str) -> tuple[str, dict[str, Any]]:
    title = ""
    author = ""
    publish_time = ""
    body_text = ""
    images: list[str] = []

    # Their SPA dumps state into window.__INITIAL_STATE__
    m = re.search(
        r"window\.__INITIAL_STATE__\s*=\s*(\{.+?\})\s*</script>", html, re.S
    )
    if m:
        raw = m.group(1)
        # XHS often emits `undefined` literals – swap those for null before json.loads.
        raw = re.sub(r"\bundefined\b", "null", raw)
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            state = None
        if state:
            note = _xhs_pick_note(state)
            if note:
                title = (note.get("title") or "").strip()
                body_text = (note.get("desc") or "").strip()
                user = note.get("user") or {}
                author = (
                    user.get("nickname") or user.get("nickName") or user.get("name") or ""
                ).strip()
                ts = note.get("time") or note.get("createTime")
                if ts:
                    publish_time = _format_ts(ts)
                for img in note.get("imageList") or []:
                    src = (
                        img.get("urlDefault")
                        or img.get("url")
                        or img.get("urlSizeLarge")
                        or img.get("urlSizeSmall")
                        or ""
                    )
                    if src:
                        images.append(src)

    if not title or not body_text:
        soup = _bs(html)
        if not title:
            og = soup.find("meta", property="og:title")
            if og and og.get("content"):
                title = og["content"].strip()
        if not body_text:
            og = soup.find("meta", property="og:description") or soup.find(
                "meta", attrs={"name": "description"}
            )
            if og and og.get("content"):
                body_text = og["content"].strip()

    metadata: dict[str, Any] = {
        "source_url": url,
        "site_type": "xiaohongshu",
        "captured_at": _now_iso(),
    }
    if title:
        metadata["title"] = title
    if author:
        metadata["author"] = author
    if publish_time:
        metadata["publish_time"] = publish_time

    parts: list[str] = []
    if title:
        parts.append(f"# {title}")
    if body_text:
        parts.append(body_text)
    for src in images:
        parts.append(f"![]({src})")
    if not parts:
        # last-ditch generic
        gen, _ = _generic_extract(url, html)
        parts.append(gen)
    return "\n\n".join(parts), metadata


def _xhs_pick_note(state: dict[str, Any]) -> dict[str, Any] | None:
    """Walk a __INITIAL_STATE__ blob looking for a note dict with title/desc."""
    # XHS state shape moves around between releases. Safest is a guarded walk.
    note_obj = state.get("note") or {}
    if isinstance(note_obj, dict):
        first_id = note_obj.get("firstNoteId") or note_obj.get("currentNoteId")
        note_detail_map = note_obj.get("noteDetailMap") or {}
        if first_id and first_id in note_detail_map:
            inner = note_detail_map[first_id] or {}
            note = inner.get("note") or inner
            if isinstance(note, dict):
                return note
        # try any value with a title
        for v in note_detail_map.values():
            if isinstance(v, dict):
                cand = v.get("note") or v
                if isinstance(cand, dict) and cand.get("title"):
                    return cand
    return None


def _format_ts(ts: Any) -> str:
    try:
        n = int(ts)
    except (TypeError, ValueError):
        return str(ts)
    if n > 10**12:  # ms
        n //= 1000
    return _dt.datetime.fromtimestamp(n, tz=_dt.timezone.utc).isoformat(timespec="seconds")


# ---- Site-specific: Bilibili --------------------------------------------------


def _bilibili_extract(url: str, html: str) -> tuple[str, dict[str, Any]]:
    soup = _bs(html)
    path = urlparse(url).path or ""

    title = ""
    author = ""
    publish_time = ""
    body_md = ""

    if path.startswith("/read/") or "read.bilibili" in url:
        # Column article
        title_node = soup.select_one("h1.title, h1.article-title")
        if title_node:
            title = title_node.get_text(strip=True)
        author_node = soup.select_one("a.up-name, .author-name")
        if author_node:
            author = author_node.get_text(strip=True)
        time_node = soup.select_one(".pub-time, .article-info time")
        if time_node:
            publish_time = time_node.get_text(strip=True)
        body_node = soup.select_one("#read-article-holder, div.article-holder, div.opus-module-content")
        if body_node is not None:
            body_md = _generic_html_to_md(body_node)
    else:
        # Video page: title + description
        title_node = soup.select_one("h1.video-title, h1[data-title], h1.tit")
        if title_node:
            title = (
                title_node.get("title")
                or title_node.get("data-title")
                or title_node.get_text(strip=True)
            )
        if not title:
            og = soup.find("meta", property="og:title")
            if og and og.get("content"):
                title = og["content"].strip()
        author_node = soup.select_one("a.up-name, a.username, .up-detail-top a")
        if author_node:
            author = author_node.get_text(strip=True)
        desc_node = soup.select_one("div.desc-info, div.video-desc, span.desc-info-text")
        if desc_node:
            body_md = desc_node.get_text("\n", strip=True)
        if not body_md:
            og = soup.find("meta", property="og:description") or soup.find(
                "meta", attrs={"name": "description"}
            )
            if og and og.get("content"):
                body_md = og["content"].strip()

    if not body_md:
        body_md, _ = _generic_extract(url, html)

    metadata: dict[str, Any] = {
        "source_url": url,
        "site_type": "bilibili",
        "captured_at": _now_iso(),
    }
    if title:
        metadata["title"] = title
    if author:
        metadata["author"] = author
    if publish_time:
        metadata["publish_time"] = publish_time

    parts = []
    if title:
        parts.append(f"# {title}")
    parts.append(body_md.strip())
    return "\n\n".join(p for p in parts if p), metadata


# ---- Generic HTML → Markdown (used by zhihu/bilibili body) -------------------


def _generic_html_to_md(node) -> str:
    """Light HTML → Markdown for already-isolated body nodes.

    Handles paragraphs, headings, lists, blockquotes, inline code, and images.
    """
    out: list[str] = []
    for el in node.children:
        if getattr(el, "name", None) is None:
            text = str(el).strip()
            if text:
                out.append(text)
            continue
        out.append(_html_block(el))
    return "\n\n".join(b for b in out if b)


def _html_block(el) -> str:
    name = el.name
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(name[1])
        return "#" * level + " " + el.get_text(strip=True)
    if name == "p":
        # preserve inline images
        images = []
        for img in el.find_all("img"):
            src = img.get("data-src") or img.get("src") or ""
            alt = img.get("alt") or ""
            if src:
                images.append(f"![{alt}]({src})")
        text = el.get_text(" ", strip=True)
        parts = [text] if text else []
        parts.extend(images)
        return "\n\n".join(parts)
    if name in ("ul", "ol"):
        marker = "-" if name == "ul" else "1."
        return "\n".join(f"{marker} {li.get_text(' ', strip=True)}" for li in el.find_all("li"))
    if name == "blockquote":
        return "> " + el.get_text(" ", strip=True).replace("\n", "\n> ")
    if name == "pre":
        return "```\n" + el.get_text("\n", strip=True) + "\n```"
    if name == "img":
        src = el.get("data-src") or el.get("src") or ""
        alt = el.get("alt") or ""
        return f"![{alt}]({src})" if src else ""
    if name in ("figure", "div", "section"):
        return _generic_html_to_md(el)
    return el.get_text(" ", strip=True)


# ---- Public entry point ------------------------------------------------------


_SITE_HANDLERS = {
    "wechat_mp": _wechat_extract,
    "zhihu": _zhihu_extract,
    "xiaohongshu": _xhs_extract,
    "bilibili": _bilibili_extract,
}


def fetch_url(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[str, dict[str, Any]]:
    """Fetch ``url`` and return (markdown_with_frontmatter, metadata).

    Args:
        url: HTTP(S) URL.
        client: Optional pre-built httpx.Client (e.g. a preconfigured client).
        timeout: Total request timeout in seconds.

    Returns:
        ``(markdown, metadata)`` where ``markdown`` already starts with a YAML
        frontmatter block built from ``metadata``.

    Raises:
        URLFetchError: when fetching or parsing fails.
    """
    if not url or not isinstance(url, str):
        raise URLFetchError("url must be a non-empty string")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise URLFetchError(f"Unsupported URL scheme: {parsed.scheme!r}")

    site_type = detect_site_type(url)
    html = _http_get(
        url, client=client, timeout=timeout, user_agent=_SITE_UA.get(site_type)
    )

    return extract_from_html(url, html, site_type=site_type)


def extract_from_html(
    url: str, html: str, *, site_type: str | None = None
) -> tuple[str, dict[str, Any]]:
    """Run the site-specific or generic HTML→markdown extractor on already-
    fetched HTML. Used by both ``fetch_url`` and the server's URL endpoint
    (after content-type sniffing).
    """
    if site_type is None:
        site_type = detect_site_type(url)

    handler = _SITE_HANDLERS.get(site_type, _generic_extract)
    body_md, metadata = handler(url, html)

    metadata.setdefault("source_url", url)
    metadata.setdefault("site_type", site_type)
    metadata.setdefault("captured_at", _now_iso())

    frontmatter = _build_frontmatter(metadata)
    return frontmatter + (body_md or ""), metadata
