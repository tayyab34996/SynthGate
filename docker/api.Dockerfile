FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY api ./api
COPY configs ./configs

RUN pip install --no-cache-dir uv && uv sync --locked --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
