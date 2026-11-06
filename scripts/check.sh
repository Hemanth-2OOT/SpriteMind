#!/usr/bin/env bash
# One command that runs everything CI would: lint, tests, README metrics and a
# short generation quality gate. Use it before a push.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON="${PYTHON:-python}"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
fi

fail=0
step() { printf '\n== %s ==\n' "$1"; }
check() {
  local name="$1"
  shift
  if "$@"; then
    printf 'ok: %s\n' "$name"
  else
    printf 'FAILED: %s\n' "$name"
    fail=1
  fi
}

step "ruff"
if command -v ruff >/dev/null 2>&1; then
  check "ruff" ruff check backend
elif command -v uvx >/dev/null 2>&1; then
  check "ruff (uvx)" uvx ruff check backend
else
  printf 'skipped: ruff not installed\n'
fi

step "pytest"
check "pytest" "$PYTHON" -m pytest

step "readme metrics"
check "readme metrics" "$PYTHON" scripts/update_readme_metrics.py --check

step "frontend build"
if command -v npm >/dev/null 2>&1 && [ -d frontend/node_modules ]; then
  check "frontend build" npm --prefix frontend run build
else
  printf 'skipped: npm or frontend/node_modules missing (run npm --prefix frontend install)\n'
fi

step "quality gate"
if [ "${SKIP_QUALITY_GATE:-0}" = "1" ]; then
  printf 'skipped: SKIP_QUALITY_GATE=1\n'
elif [ -f models/checkpoints/svan_checkpoint_best.pt ]; then
  check "quality gate" "$PYTHON" scripts/quality_gate.py --limit "${QUALITY_GATE_SAMPLES:-8}"
else
  printf 'skipped: no trained checkpoint\n'
fi

if [ "$fail" -ne 0 ]; then
  printf '\nCHECK FAILED\n'
  exit 1
fi
printf '\nALL CHECKS PASSED\n'
