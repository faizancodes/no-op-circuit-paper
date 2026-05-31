#!/usr/bin/env bash
# Sanity-check an arXiv source bundle: main .tex present, all \includegraphics
# targets exist, bibliography available (refs.bib or matching .bbl), the document
# compiles, and the log is free of serious errors. Overfull boxes are reported
# but do not fail the check unless they are extreme.
#
# Usage:
#   scripts/check_arxiv_bundle.sh [path/to/main.tex]
# Default target: arxiv_upload/main.tex if present, else paper/draft.tex.
#
# Note: this only INSPECTS/COMPILES an existing bundle/source. It never adds
# activation caches, *.pt/*.pth/*.safetensors, Modal artifacts, or large result
# directories to anything (bundle staging lives in scripts/make_arxiv_package.sh,
# which excludes those).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- locate the main .tex --------------------------------------------------
MAIN="${1:-}"
if [ -z "$MAIN" ]; then
  if   [ -f "$REPO/arxiv_upload/main.tex" ]; then MAIN="$REPO/arxiv_upload/main.tex"
  elif [ -f "$REPO/paper/draft.tex"       ]; then MAIN="$REPO/paper/draft.tex"
  else echo "ERROR: no main .tex given and neither arxiv_upload/main.tex nor paper/draft.tex exists"; exit 1
  fi
fi
# (a) fail on missing main .tex
if [ ! -f "$MAIN" ]; then echo "ERROR: main .tex not found: $MAIN"; exit 1; fi
DIR="$(cd "$(dirname "$MAIN")" && pwd)"
BASE="$(basename "$MAIN" .tex)"
echo "[check] main tex : $DIR/$BASE.tex"
cd "$DIR"

fail=0

# (b) every \includegraphics target must exist on disk -----------------------
# Image paths in this project are always {figures/<name>.<ext>}; match those
# robustly (the optional [..] option block can contain alt= braces, so we key on
# the path prefix rather than brace-parsing).
echo "[check] figures referenced by \\includegraphics:"
n_inc="$(grep -oE '\\includegraphics' "$BASE.tex" | wc -l | tr -d ' ')"
# Portable (bash 3.2): build array via a read loop, not mapfile.
figs=()
while IFS= read -r line; do
  [ -n "$line" ] && figs+=("$line")
done < <(grep -oE 'figures/[A-Za-z0-9_./-]+\.(png|pdf|jpg|jpeg)' "$BASE.tex" | sort -u || true)
n_figs="${#figs[@]}"
if [ "$n_figs" -eq 0 ]; then
  echo "  (no figures/ paths found; \\includegraphics count = $n_inc)"
else
  for f in "${figs[@]}"; do
    if [ -f "$f" ]; then
      echo "  ok      $f"
    else
      echo "  MISSING $f"; fail=1
    fi
  done
fi
echo "  (\\includegraphics calls: $n_inc; distinct figure targets: $n_figs)"

# (c) bibliography: refs.bib OR a .bbl matching the main basename ------------
if [ -f "refs.bib" ]; then
  echo "[check] bibliography: refs.bib present"
elif [ -f "$BASE.bbl" ]; then
  echo "[check] bibliography: $BASE.bbl present (pre-generated)"
else
  echo "ERROR: neither refs.bib nor $BASE.bbl found next to $BASE.tex"; fail=1
fi

# (d) compile in an isolated temp copy (latexmk if available, else pdflatex/
#     bibtex fallback). Building out-of-place keeps the bundle pristine AND
#     proves the bundle is self-contained.
echo "[check] compiling $BASE.tex (isolated build) ..."
TMP="$(mktemp -d "${TMPDIR:-/tmp}/arxivchk.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
cp "$BASE.tex" "$TMP/"
[ -f refs.bib ]    && cp refs.bib    "$TMP/" || true
[ -f "$BASE.bbl" ] && cp "$BASE.bbl" "$TMP/" || true
[ -d figures ]     && cp -R figures  "$TMP/" || true
cp ./*.sty ./*.cls "$TMP/" 2>/dev/null || true
set +e
( cd "$TMP"
  if command -v latexmk >/dev/null 2>&1; then
    latexmk -pdf -interaction=nonstopmode "$BASE.tex" >/dev/null 2>&1
  else
    echo "  (latexmk not found; falling back to pdflatex -> bibtex -> pdflatex x2)"
    pdflatex -interaction=nonstopmode "$BASE.tex" >/dev/null 2>&1
    bibtex   "$BASE"                              >/dev/null 2>&1
    pdflatex -interaction=nonstopmode "$BASE.tex" >/dev/null 2>&1
    pdflatex -interaction=nonstopmode "$BASE.tex" >/dev/null 2>&1
  fi )
build_rc=$?
set -e
LOG="$TMP/$BASE.log"
if [ ! -f "$LOG" ]; then echo "ERROR: no log produced ($LOG); build_rc=$build_rc"; exit 1; fi
echo "[check] build exit code: $build_rc; PDF: $([ -f "$TMP/$BASE.pdf" ] && echo "$BASE.pdf present" || echo "MISSING")"
[ -f "$TMP/$BASE.pdf" ] || fail=1

# (e) serious errors in the log ---------------------------------------------
echo "[check] scanning $LOG for serious errors:"
patterns='LaTeX Error|Undefined control sequence|Citation .* undefined|Reference .* undefined|File .* not found|Emergency stop|Fatal error'
if grep -aE "$patterns" "$LOG" >/dev/null 2>&1; then
  echo "  SERIOUS ISSUES FOUND:"
  grep -anE "$patterns" "$LOG" | sed 's/^/    /' | head -40
  fail=1
else
  echo "  none."
fi

# (f) overfull boxes: report, fail only if extreme (> 150pt) -----------------
n_hbox="$(grep -ac 'Overfull \\hbox' "$LOG" || true)"
n_vbox="$(grep -ac 'Overfull \\vbox' "$LOG" || true)"
maxpt="$(grep -aoE 'Overfull \\[hv]box \(([0-9]+\.[0-9]+)pt' "$LOG" 2>/dev/null \
          | grep -oE '[0-9]+\.[0-9]+' | sort -rn | head -1 || true)"
maxpt="${maxpt:-0}"
echo "[check] overfull boxes: hbox=$n_hbox vbox=$n_vbox max=${maxpt}pt (warnings only, not a failure unless extreme)"
if awk "BEGIN{exit !($maxpt > 150)}"; then
  echo "  WARNING: an overfull box exceeds 150pt (${maxpt}pt) -- inspect the PDF; failing."
  fail=1
fi

# --- verdict ----------------------------------------------------------------
if [ "$fail" -ne 0 ]; then
  echo "[check] RESULT: FAIL"
  exit 1
fi
echo "[check] RESULT: PASS"
