#!/usr/bin/env bash
# Compile the Overleaf report locally WITHOUT polluting the report/ submodule.
#
# LaTeX scatters build files (.aux .log .bbl .pdf ...) next to the source by
# default. report/ is a git submodule mirroring the Overleaf project, so those
# artifacts would show up as dirty/untracked and could get pushed to Overleaf.
# This script redirects every artifact to build/report/ (gitignored in the main
# repo) instead, leaving the submodule pristine.
#
# Usage:  ./compile_report.sh         build, then report reference warnings
#         ./compile_report.sh open    also open the PDF (if a viewer exists)
#
# Needs pdflatex + bibtex (TeX Live). latexmk not required.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/report"
OUT="$ROOT/build/report"
mkdir -p "$OUT"

# bibtex runs next to the .aux (in OUT) but must find main.bib + the .bst in SRC.
export BIBINPUTS="$SRC:" BSTINPUTS="$SRC:"

pdf() {
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT" main.tex >/dev/null 2>&1
}

cd "$SRC"
echo "[1/4] pdflatex"; pdf || { echo "  pdflatex failed - see $OUT/main.log"; exit 1; }
echo "[2/4] bibtex";   ( cd "$OUT" && bibtex main >/dev/null 2>&1 ) || echo "  (bibtex reported problems - see $OUT/main.blg)"
echo "[3/4] pdflatex"; pdf || { echo "  pdflatex failed - see $OUT/main.log"; exit 1; }
echo "[4/4] pdflatex"; pdf || { echo "  pdflatex failed - see $OUT/main.log"; exit 1; }

echo
echo "PDF -> $OUT/main.pdf"

warn=$(grep -nE "Citation .* undefined|Reference .* undefined|multiply defined" "$OUT/main.log" || true)
if [ -n "$warn" ]; then
  echo
  echo "Reference warnings:"
  printf '%s\n' "$warn" | sed 's/^/  /'
fi

if [ "${1:-}" = "open" ]; then
  command -v xdg-open >/dev/null 2>&1 && xdg-open "$OUT/main.pdf" >/dev/null 2>&1 || echo "(no xdg-open; PDF is at $OUT/main.pdf)"
fi
