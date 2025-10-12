#!/bin/bash
# Unified determinism environment for local and CI
# Source this script before running tests or audits

export PYTHONHASHSEED=0
export TZ=UTC
export MPLBACKEND=Agg
export SOURCE_DATE_EPOCH=1577836800

# Single-threaded execution for determinism
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

# GlassAlpha-specific
export GLASSALPHA_NO_PROGRESS=1
export GLASSALPHA_TELEMETRY=0
export GLASSALPHA_DETERMINISTIC=1

# Locale
export LC_ALL=C

echo "✅ Determinism environment configured"
echo "  - Single-threaded: OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  - Fixed timezone: TZ=${TZ}"
echo "  - Headless matplotlib: MPLBACKEND=${MPLBACKEND}"
