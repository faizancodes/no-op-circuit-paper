# arXiv source-bundle checklist

The bundle is produced reproducibly by `scripts/make_arxiv_package.sh`
(stages into `arxiv_upload/`, emits `arxiv_upload/noop_circuit_arxiv_source.tar.gz`).
This file documents what that bundle must and must not contain, and the
sanity checks to run before uploading.

## Must include

- The main `.tex` file (we ship `paper/draft.tex` renamed to `main.tex`).
- The matching pre-generated bibliography `.bbl`. **Its basename must match the
  main TeX file** — we ship `paper/draft.bbl` renamed to `main.bbl` so arXiv can
  typeset the bibliography without re-running BibTeX.
- `refs.bib` (so arXiv can re-run BibTeX if it prefers).
- Every figure referenced by `\includegraphics` (this paper: 23 PNGs under
  `figures/`).
- Any custom `.sty`, `.cls`, or header fragment actually required by the main
  TeX. **This paper needs none** — it uses the standard `article` class plus
  CTAN packages only (the `paper/article_header.tex` fragment is inlined into
  `draft.tex` at build time, not a separate file at submission).

## Must exclude

- Activation caches and any `results/` tensors.
- Model/tensor files: `.pt`, `.pth`, `.safetensors`, `.bin`.
- Large metric/JSON dumps.
- Modal logs and any run logs.
- Temporary LaTeX build files: `.aux`, `.out`, `.log`, `.fls`, `.fdb_latexmk`,
  `.synctex.gz`, `.toc`, `.blg`.
- Previously generated PDFs (unless intentionally submitting the compiled PDF
  separately as the paper PDF — not part of the *source* bundle).
- Unused figures, editor backups, and macOS resource forks (`._*`, `.DS_Store`).

## Build the bundle

```bash
# rebuild the paper first, then stage + tar + verify:
BUILD=1 bash scripts/make_arxiv_package.sh
```

## Sanity checks

List the bundle contents (should be only `main.tex`, `main.bbl`, `refs.bib`,
`figures/…`):

```bash
tar -tzf arxiv_upload/noop_circuit_arxiv_source.tar.gz | sort
```

Confirm no excluded artifacts slipped in (should print nothing):

```bash
tar -tzf arxiv_upload/noop_circuit_arxiv_source.tar.gz \
  | grep -Ei '\.(pt|pth|safetensors|bin|aux|log|out|toc|blg|fls|fdb_latexmk|synctex\.gz)$|results/|cache|/\._'
```

Check that every figure referenced actually exists (compare the
`\includegraphics` targets against files on disk):

```bash
rg -o 'figures/[A-Za-z0-9_./-]+\.(png|pdf|jpg|jpeg)' paper/draft.tex | sort -u \
  | while read -r f; do [ -f "paper/$f" ] || echo "MISSING: $f"; done
```

`scripts/make_arxiv_package.sh` runs the missing-figure check automatically and
aborts if any `\includegraphics` target is absent.

## Clean compile test (mirrors arXiv's autotex)

```bash
mkdir -p /tmp/arxiv_test && tar -xzf arxiv_upload/noop_circuit_arxiv_source.tar.gz -C /tmp/arxiv_test
cd /tmp/arxiv_test
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
