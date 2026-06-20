#!/bin/bash
# GAAL v3 — Cron-compatible wrapper for Hermes cron jobs
# Called by Hermes cron: cronjob(..., skills=["gaal-v3"], ...)
# Usage: gaal-cron.sh <goal> [mode] [max_loops]

set -euo pipefail

GAAL_ROOT="$HOME/.hermes/gaal_v3"
cd "$GAAL_ROOT"

GOAL="${1:-设计一个简单的文件备份系统}"
MODE="${2:-lite}"
MAX_LOOPS="${3:-4}"

export GAAL_V3_ROOT="$GAAL_ROOT"

exec python -m gaal_v3.run "$GOAL" "$MODE" "$MAX_LOOPS" 2>&1
