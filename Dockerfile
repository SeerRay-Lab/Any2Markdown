# Any2Markdown container image.
#
# Build:
#   docker build -t any2markdown .
#
# Optional build args (all default to the public/official value):
#   PYTHON_IMAGE      base image, e.g. docker.m.daocloud.io/library/python:3.12-slim
#   DEBIAN_MIRROR     apt mirror host, e.g. mirrors.tuna.tsinghua.edu.cn
#   PIP_INDEX_URL     PyPI index, e.g. https://pypi.tuna.tsinghua.edu.cn/simple
#   INSTALL_EDITABLE  "true" to install -e (bind-mount dev workflow)
#
# Example for a fast build in China:
#   docker build \
#     --build-arg PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.12-slim \
#     --build-arg DEBIAN_MIRROR=mirrors.tuna.tsinghua.edu.cn \
#     --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#     -t any2markdown .

ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ARG DEBIAN_MIRROR=
ARG PIP_INDEX_URL=
ARG INSTALL_EDITABLE=false

# PLAYWRIGHT_BROWSERS_PATH keeps Chromium in an image-level directory so the
# non-root runtime user can launch it without downloading another copy.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# ---- System packages --------------------------------------------------------
# The official base image is used by default. Setting DEBIAN_MIRROR rewrites
# the sources first, so builds in China stay fast without hard-coding a mirror
# for everyone else.
#
# Installed:
#   - minimal headless LibreOffice for DOC/PPT/XLSX → PDF
#   - CJK + Latin fonts so rendered pages keep their glyphs
#   - rendering/runtime libs used by LibreOffice and Chromium
# poppler-utils is intentionally absent: PDF rendering goes through PyMuPDF.
# Debian's per-package copyright files under /usr/share/doc are kept because
# the image redistributes LibreOffice (MPL-2.0) and fonts (OFL); other docs
# are removed to save space.
RUN if [ -n "$DEBIAN_MIRROR" ]; then \
        sed -i "s|deb.debian.org|$DEBIAN_MIRROR|g; s|security.debian.org|$DEBIAN_MIRROR|g" \
            /etc/apt/sources.list.d/debian.sources 2>/dev/null || \
        sed -i "s|deb.debian.org|$DEBIAN_MIRROR|g; s|security.debian.org|$DEBIAN_MIRROR|g" \
            /etc/apt/sources.list 2>/dev/null || true; \
    fi \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-writer libreoffice-calc libreoffice-impress \
        fonts-noto-cjk fonts-noto fonts-dejavu-core \
        libgl1 libxrender1 libxext6 libsm6 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
              /usr/share/man/* \
    && find /usr/share/doc -type f ! -name copyright -delete \
    && find /usr/share/doc -type d -empty -delete

# ---- Runtime user -----------------------------------------------------------
# uid 1000 matches the common desktop user, so the README's
# "-v $(pwd)/data:/app/data" mount is writable out of the box on most hosts.
# If your host user has another uid, either chown the mounted directory or run
# the container with --user "$(id -u):$(id -g)".
RUN useradd --uid 1000 --user-group --create-home --home-dir /home/app \
            --shell /usr/sbin/nologin app \
    && mkdir -p /app/data/media \
    && chown -R app:app /app

# ---- Python package ---------------------------------------------------------
# Copy metadata and sources together: the build backend needs the package tree,
# so a split "deps first" layer would not help here.
COPY --chown=app:app pyproject.toml any2md.yaml README.md LICENSE ./
COPY --chown=app:app any2markdown ./any2markdown

RUN if [ -n "$PIP_INDEX_URL" ]; then \
        pip config set --global global.index-url "$PIP_INDEX_URL"; \
    fi \
    && if [ "$INSTALL_EDITABLE" = "true" ]; then \
        pip install -e .; \
    else \
        pip install .; \
    fi \
    && python -c "import importlib.metadata as m, pymupdf4llm; \
print('pymupdf', m.version('pymupdf'), '| pymupdf4llm', m.version('pymupdf4llm')); \
assert callable(getattr(pymupdf4llm, 'to_markdown', None)), \
    'pymupdf4llm.to_markdown missing; check the pymupdf/pymupdf4llm pins in pyproject.toml'"

# ---- Headless browser -------------------------------------------------------
# --with-deps installs Chromium's shared libraries. Keep the chown in the same
# layer as the download so the browser files are not duplicated in a copy-up.
RUN playwright install --with-deps chromium \
    && chown -R app:app /ms-playwright \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# ---- Runtime ----------------------------------------------------------------
ENV HOME=/home/app
# Persist the web-UI model profiles and downloaded images under the same
# directory the README already mounts: -v "$(pwd)/data:/app/data".
ENV ANY2MD_PROFILES_FILE=/app/data/model_profiles.json \
    ANY2MD_MEDIA_DIR=/app/data/media

USER app

EXPOSE 8088

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8088/health || exit 1

CMD ["uvicorn", "any2markdown.server:app", "--host", "0.0.0.0", "--port", "8088"]
