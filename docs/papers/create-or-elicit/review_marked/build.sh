#!/usr/bin/env bash
# Build paper/main.pdf with tectonic (static binary; ~/tools/tectonic/tectonic or on PATH).
# Regenerate figures first:  python3 paper/scripts/figures.py
set -e
cd "$(dirname "$0")"
T=${TECTONIC:-$HOME/tools/tectonic/tectonic}
command -v "$T" >/dev/null 2>&1 || T=tectonic
"$T" -X compile main.tex 2>&1 | grep -v '^note: downloading' | grep -E 'error|Error|undefined|Output|warning: .*Overfull' || true
ls -la main.pdf
