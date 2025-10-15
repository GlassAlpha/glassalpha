# syntax=docker/dockerfile:1.7

# Build stage
FROM python:3.11-slim@sha256:ad48727987b259854d52241feff7d59fbab2ea7e233bddc77968a9c9b942d9c4 AS build

LABEL org.opencontainers.image.title="GlassAlpha"
LABEL org.opencontainers.image.description="AI Compliance Toolkit - transparent, auditable, regulator-ready ML audits"
LABEL org.opencontainers.image.authors="GlassAlpha Team"
LABEL org.opencontainers.image.license="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/glassalpha/glassalpha"

# Set build environment
ENV PYTHONHASHSEED=42 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only what's needed for build
COPY pyproject.toml README.md MANIFEST.in ./
COPY src ./src

# Build wheel
RUN pip install --upgrade pip==24.2 build==1.2.1 && \
    python -m build --wheel

# Runtime stage
FROM python:3.11-slim@sha256:ad48727987b259854d52241feff7d59fbab2ea7e233bddc77968a9c9b942d9c4

# Runtime environment
ENV PYTHONHASHSEED=42 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    shared-mime-info \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 glassalpha && \
    mkdir -p /data /output && \
    chown -R glassalpha:glassalpha /data /output

USER glassalpha
WORKDIR /home/glassalpha

# Copy wheel from build stage
COPY --from=build --chown=glassalpha:glassalpha /build/dist/*.whl /tmp/

# Install wheel
RUN pip install --user /tmp/*.whl && \
    rm /tmp/*.whl

# Add user site-packages to PATH
ENV PATH="/home/glassalpha/.local/bin:${PATH}"

# Set volumes
VOLUME ["/data", "/output"]
WORKDIR /data

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD glassalpha --version || exit 1

ENTRYPOINT ["glassalpha"]
CMD ["--help"]
