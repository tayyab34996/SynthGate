FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY configs ./configs
COPY artifacts ./artifacts

RUN pip install --no-cache-dir uv && uv sync --locked --no-dev

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "dashboard/app.py", "--server.address", "0.0.0.0"]
