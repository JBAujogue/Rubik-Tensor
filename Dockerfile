
FROM python:3.11-slim-bookworm AS base

# --- Build stage ---

FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:0.8.3 / uv / uvx / bin/

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

COPY uv.lock pyproject.toml /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
    --frozen \
    --extra torch-cpu \
    --no-dev \
    --no-install-project

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
    --frozen \
    --extra torch-cpu \
    --no-dev

# --- Final stage ---

FROM base AS final

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

ENTRYPOINT [ "python", "-m", "rubik", "interface" ]

CMD []
