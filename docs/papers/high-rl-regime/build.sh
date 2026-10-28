#!/usr/bin/env bash
# Build main.pdf with tectonic (a self-contained LaTeX engine; on serrano at /scratch/cowifornia/tools/tectonic).
# Usage: bash build.sh            (notes on, as committed)
# The first build downloads tectonic's TeX bundle into TECTONIC_CACHE_DIR (a few hundred MB, kept on /scratch).
set -euo pipefail
cd "$(dirname "$0")"
T=${TECTONIC:-/scratch/cowifornia/tools/tectonic/tectonic}
command -v "$T" >/dev/null 2>&1 || T=tectonic
export TECTONIC_CACHE_DIR=${TECTONIC_CACHE_DIR:-/scratch/cowifornia/tools/tectonic-cache}
# references.bib is generated from Zotero at the repo root; copy it rather than editing or forking it
cp ../../../references.bib references.bib
"$T" --keep-logs main.tex 2>&1 | grep -v '^note: downloading' | grep -vE '^(note|warning): .*(Underfull|Overfull)' || true
ls -la main.pdf
