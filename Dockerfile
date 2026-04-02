# Use Python 3.12-slim as the base image
FROM python:3.12-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Set environment variables
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Synchronize everything (installs dependencies from uv.lock)
RUN uv sync --frozen

# Use the virtualenv's uvicorn module directly
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
