#!/usr/bin/env bash
# Build a clean arXiv source package from the generated paper sources.
#
# Stages ONLY what arXiv needs:
#   - main.tex   (= paper/draft.tex, the pandoc-generated canonical LaTeX)
#   - main.bbl   (= paper/draft.bbl, so arXiv need not re-run bibtex)
#   - refs.bib   (so arXiv CAN re-run bibtex if it prefers)
#   - figures/   (ONLY the files referenced by \includegraphics)
#   - any local .sty/.cls actually \usepackage'd by main.tex (none for this paper)
#
# Excludes (never staged): results/, activation/tensor caches, *.pt/*.pth/*.bin,
# large JSON caches, logs, prior PDFs, unused figures, and the LaTeX build
# by-products *.aux/*.log/*.out/*.toc/*.blg/*.fls/*.fdb_latexmk.
#
# Also: writes MANIFEST.txt, verifies every \includegraphics target exists, and
# emits a .tar.gz with main.tex at the archive ROOT (no top-level folder).
#
# Usage:
#   bash scripts/make_arxiv_package.sh            # stage into arxiv_upload/
#   BUILD=1 bash scripts/make_arxiv_package.sh    # rebuild the paper first
#   bash scripts/make_arxiv_package.sh build/arxiv  # custom staging dir
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TEX_SRC="paper/draft.tex"
BBL_SRC="paper/draft.bbl"
BIB_SRC="paper/refs.bib"
FIG_ROOT="paper"                 # \includegraphics{figures/...} is relative to paper/
OUT_DIR="${1:-arxiv_upload}"     # clean staging directory
TARBALL="noop_circuit_arxiv_source.tar.gz"

# 0. (Re)build the paper if asked, or if the generated TeX is missing.
if [[ "${BUILD:-0}" == "1" || ! -f "$TEX_SRC" ]]; then
  echo "[arxiv] running paper/build.sh first (BUILD=${BUILD:-0}, tex present: $([[ -f $TEX_SRC ]] && echo yes || echo no))"
  bash paper/build.sh
fi
for f in "$TEX_SRC" "$BBL_SRC" "$BIB_SRC"; do
  [[ -f "$f" ]] || { echo "ERROR: required source $f not found (run: bash paper/build.sh)"; exit 1; }
done

# 1. Fresh staging directory.
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/figures"

# 2. Copy required sources, renaming to main.* so arXiv picks the right main file.
cp "$TEX_SRC" "$OUT_DIR/main.tex"
cp "$BBL_SRC" "$OUT_DIR/main.bbl"
cp "$BIB_SRC" "$OUT_DIR/refs.bib"

# 3. Copy ONLY referenced figures; fail if any \includegraphics target is missing.
python3 - "$OUT_DIR/main.tex" "$FIG_ROOT" "$OUT_DIR" <<'PY'
import re, shutil, sys
from pathlib import Path
tex_path, fig_root, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tex = tex_path.read_text(encoding="utf-8")
# Every figure include in this paper is {figures/<name>.<ext>}; this is robust to
# multi-option \includegraphics[...] blocks (incl. alt= text containing ']').
refs = sorted(set(re.findall(r"figures/[A-Za-z0-9_./-]+\.(?:png|pdf|jpg|jpeg)", tex)))
n_inc = len(re.findall(r"\\includegraphics", tex))
copied, missing = [], []
for r in refs:
    src = fig_root / r
    if not src.exists():
        missing.append(r); continue
    dst = out_dir / r
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst); copied.append(r)
print(f"[arxiv] \\includegraphics calls: {n_inc}; distinct figure targets: {len(refs)}; "
      f"copied: {len(copied)}; missing: {len(missing)}")
if missing:
    print("ERROR: referenced figures not found on disk:", missing); sys.exit(1)
if n_inc != len(refs):
    print(f"WARNING: {n_inc} \\includegraphics calls but {len(refs)} distinct targets "
          f"(reused figures are fine; investigate only if unexpected).")
PY

# 4. Stage any local .sty/.cls that main.tex actually \usepackage's (none expected).
shopt -s nullglob
for sty in paper/*.sty paper/*.cls; do
  base="$(basename "${sty%.*}")"
  if grep -Eq "\\\\(usepackage|documentclass)(\[[^]]*\])?\{[^}]*${base}[^}]*\}" "$OUT_DIR/main.tex"; then
    cp "$sty" "$OUT_DIR/"; echo "[arxiv] staged local style: $(basename "$sty")"
  fi
done
shopt -u nullglob

# 5. Manifest of staged files (excluding the tarball/manifest themselves).
( cd "$OUT_DIR" && find . -type f ! -name "$TARBALL" ! -name MANIFEST.txt | sort ) > "$OUT_DIR/MANIFEST.txt"
echo "[arxiv] wrote $OUT_DIR/MANIFEST.txt ($(wc -l < "$OUT_DIR/MANIFEST.txt" | tr -d ' ') files)"

# 6. Tarball with main.tex at the archive root; strip macOS resource forks.
#    Member list = required sources + figures + any staged local styles.
MEMBERS=(main.tex main.bbl refs.bib figures)
shopt -s nullglob
for s in "$OUT_DIR"/*.sty "$OUT_DIR"/*.cls; do MEMBERS+=("$(basename "$s")"); done
shopt -u nullglob
( cd "$OUT_DIR" && COPYFILE_DISABLE=1 tar --no-mac-metadata -czf "$TARBALL" "${MEMBERS[@]}" 2>/dev/null \
  || COPYFILE_DISABLE=1 tar -czf "$TARBALL" "${MEMBERS[@]}" )
echo "[arxiv] wrote $OUT_DIR/$TARBALL"

# 7. Forbidden-artifact guard: the tarball must contain no caches/logs/build junk,
#    tensors, virtualenvs, VCS, or editor/OS cruft.
echo "[arxiv] forbidden-artifact check:"
if tar -tzf "$OUT_DIR/$TARBALL" | grep -Ei '\.(pt|pth|safetensors|bin|ckpt|aux|log|out|toc|blg|fls|fdb_latexmk|synctex\.gz)$|(^|/)(results|wandb|__pycache__|\.venv|\.git|node_modules)/|cache|/\._|/\.DS_Store'; then
  echo "ERROR: forbidden artifacts found in tarball (see above)"; exit 1
fi
echo "  clean."
echo "[arxiv] package ready: $OUT_DIR/$TARBALL"
