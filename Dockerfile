# jennit back-end: FastAPI batch API + analysis pipeline (single worker).
# The web UI lives in the separate front-end container (see frontend/).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# imagemagick: `identify` (JPEG quality analysis)
# libglib2.0-0: required by OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        imagemagick \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY jennit/ ./jennit/

# headless OpenCV build for the container (no GUI/X11 dependencies)
RUN sed 's/^opencv-python$/opencv-python-headless/' requirements.txt \
      > /tmp/requirements.docker.txt \
    && pip install --no-cache-dir -r /tmp/requirements.docker.txt \
    && rm /tmp/requirements.docker.txt \
    && pip install --no-cache-dir --no-deps .

# unprivileged user; /spool and /data are the volume mount points
RUN useradd --create-home --shell /usr/sbin/nologin --uid 10001 jennit \
    && mkdir -p /spool /data \
    && chown -R jennit:jennit /spool /data
USER jennit

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=12 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["python", "-m", "jennit.server"]
