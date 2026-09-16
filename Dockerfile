# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — builder
#
# Dependencies are compiled into a self-contained virtualenv here so the final
# image never carries pip, wheels or build toolchains. Only /opt/venv crosses
# the stage boundary.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copied on its own so this layer is reused whenever only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# fonts-dejavu-core provides the DejaVuSans TTFs that the PDF export probes for.
# Without them reportlab falls back to Helvetica and Macedonian Cyrillic renders
# as empty boxes. curl is used by the container healthcheck below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core curl \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged runtime user, created before the COPYs so ownership is set once.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser scripts/ ./scripts/

# The app writes user uploads here and mounts it at /static/uploads. Created
# up front with the right owner so it also works when a volume is mounted over it.
RUN mkdir -p /app/uploads && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
