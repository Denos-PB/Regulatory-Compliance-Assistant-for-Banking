FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:python3.12-trixie-slim /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra observability
COPY src/ src/
COPY config.yaml .
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]