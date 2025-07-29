FROM python:3.11-slim-bookworm

# Get uv from distro-less image
COPY --from=ghcr.io/astral-sh/uv:0.8.3 / uv / uvx / bin/

ADD . /app

WORKDIR /app

RUN uv sync --locked --extra torch-cpu

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 7860

ENTRYPOINT [ "python", "-m", "rubik", "interface" ]
CMD []
