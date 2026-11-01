#!/usr/bin/env bash
# Build main.pdf with tectonic (a self-contained LaTeX engine; on serrano at /scratch/cowifornia/tools/tectonic).
# Usage: bash build.sh
# Writes main.pdf (notes on, as committed) and main_clean.pdf (the same paper with \notesfalse: no talk boxes, TODOs
# or candidate figures).
# The first build downloads tectonic's TeX bundle into TECTONIC_CACHE_DIR (a few hundred MB, kept on /scratch).
set -euo pipefail
cd "$(dirname "$0")"
T=${TECTONIC:-/scratch/cowifornia/tools/tectonic/tectonic}
command -v "$T" >/dev/null 2>&1 || T=tectonic
export TECTONIC_CACHE_DIR=${TECTONIC_CACHE_DIR:-/scratch/cowifornia/tools/tectonic-cache}
# references.bib is generated from Zotero at the repo root; copy it rather than editing or forking it
cp ../../../references.bib references.bib
# Papers not yet in Zotero wait in to_zotero/ (see its README); cite them meanwhile from the build copy only.
for f in to_zotero/*.bib; do
  key=$(sed -n 's/^@[A-Za-z]*{\([^,]*\),.*/\1/p' "$f" | head -1)
  grep -q "{$key," references.bib || { echo; cat "$f"; } >> references.bib
done
build() { "$T" --keep-logs "$1" 2>&1 | grep -v '^note: downloading' | grep -vE '^(note|warning): .*(Underfull|Overfull)' || true; }
build main.tex
sed 's/^\\notestrue$/\\notesfalse/' main.tex > main_clean.tex
grep -q '^\\notesfalse$' main_clean.tex || { echo "build.sh: no \\notestrue line in main.tex to switch off" >&2; exit 1; }
build main_clean.tex
rm -f main_clean.tex main_clean.log main_clean.blg
ls -la main.pdf main_clean.pdf
