# Multi-stage build for Email Triage Environment
FROM python:3.12-slim AS builder

WORKDIR /app
COPY . /app/env
WORKDIR /app/env

# Install uv securely
RUN pip install uv

# Create virtual environment and install dependencies natively
RUN uv sync --frozen --no-install-project --no-editable

# Final runtime stage
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/env/.venv /app/.venv

# Copy the environment code
COPY --from=builder /app/env /app/env

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
