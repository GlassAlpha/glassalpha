FROM python:3.11-slim

LABEL org.opencontainers.image.title="GlassAlpha"
LABEL org.opencontainers.image.description="AI Compliance Toolkit - transparent, auditable, regulator-ready ML audits"
LABEL org.opencontainers.image.authors="GlassAlpha Team"
LABEL org.opencontainers.image.license="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/glassalpha/glassalpha"

# Set environment for determinism and non-interactive installs
ENV PYTHONHASHSEED=42 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for WeasyPrint and PDF generation
# These provide consistent PDF rendering across environments
RUN apt-get update && apt-get install -y --no-install-recommends \
    # WeasyPrint dependencies
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    # Standard fonts for consistent rendering
    fonts-liberation \
    fonts-dejavu-core \
    fonts-freefont-ttf \
    # Build tools for Python packages
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 glassalpha && \
    mkdir -p /app /data /output && \
    chown -R glassalpha:glassalpha /app /data /output

# Set working directory
WORKDIR /app

# Copy package files
COPY --chown=glassalpha:glassalpha . /app/

# Install GlassAlpha with all optional dependencies
# Use --no-cache-dir to keep image size down
USER glassalpha
RUN pip install --user --no-cache-dir -e ".[explain,viz,docs]"

# Add user site-packages to PATH
ENV PATH="/home/glassalpha/.local/bin:${PATH}"

# Set data and output directories as volumes
VOLUME ["/data", "/output"]

# Default working directory for user data
WORKDIR /data

# Health check: verify CLI is accessible
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD glassalpha --version || exit 1

# Entry point: GlassAlpha CLI
ENTRYPOINT ["glassalpha"]
CMD ["--help"]
