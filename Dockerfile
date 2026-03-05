FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy package
COPY pyproject.toml README.md ./
COPY src/ src/

# Install package (system-wide)
RUN uv pip install --system --no-cache .

# Default command
CMD ["jackknify", "--help"]