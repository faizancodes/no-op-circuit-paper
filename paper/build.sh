#!/usr/bin/env bash
# Build paper/draft.pdf using a vanilla article 11pt a4paper preamble that
# matches the visual style of typical LaTeX submissions (Computer Modern,
# 1in margins, colored hyperref links, multi-line author block).
#
# Pipeline:
#   0. Rewrite draft.md → .draft_article.md by stripping the markdown
#      title + the centered \begin{center} author block + the
#      "## Abstract" heading + abstract body, injecting them as pandoc
#      YAML frontmatter so \maketitle / \begin{abstract} render from the
#      vanilla \maketitle layout.
#   1. pandoc body+metadata → tex via the default standalone template
#      with `--include-in-header=paper/article_header.tex` and class
#      options for article 11pt a4paper.
#   2. unicode → LaTeX substitution + bibliography injection (mirrors
#      build_neurips.sh).
#   3. pdflatex × 2 → bibtex → pdflatex × 2.
#
# Output: paper/draft.pdf
#
# For a NeurIPS-style submission build, use paper/build_neurips.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d "$ROOT/.venv" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
export PATH="/Library/TeX/texbin:$PATH"

# 0. Rewrite draft.md → .draft_article.md with YAML frontmatter
python3 - <<'PY'
import re
from pathlib import Path

src = Path("paper/draft.md").read_text(encoding="utf-8")

m_title = re.match(r"^# (.+?)\n", src)
assert m_title, "could not parse title from line 1 of draft.md"
title = m_title.group(1).strip()

m_abs = re.search(r"##\s*Abstract\s*\n+(.+?)\n+##\s+1\.", src, re.S)
assert m_abs, "could not extract abstract text"
abstract_md = m_abs.group(1).strip()

stripped = src
stripped = re.sub(r"^# .+?\n", "", stripped, count=1)
stripped = re.sub(r"\\begin\{center\}.*?\\end\{center\}\s*\n", "",
                  stripped, count=1, flags=re.S)
stripped = re.sub(r"##\s*Abstract.*?\n##\s+1\.", "## 1.",
                  stripped, count=1, flags=re.S)

# Body-heading promotion: `## N. Title` → `# Title` so pandoc emits
# \section{} (proper top-level numbering instead of \subsection{}).
# Mirrors build_neurips.sh — only touch body, leave appendix alone.
parts = re.split(r"(\\newpage\s*\n\\appendix)", stripped, maxsplit=1, flags=re.M)
body, appendix_marker, appendix_body = (parts + ["", "", ""])[:3]
appendix_tail = appendix_marker + appendix_body
body = re.sub(r"^## \d+\. +", "# ", body, flags=re.M)
body = re.sub(r"^### \d+\.\d+ +", "## ", body, flags=re.M)
body = re.sub(r"^## References\s*$", "# References", body, flags=re.M)
stripped = body + appendix_tail

title_yaml = title.replace('"', r'\"')
abs_yaml_lines = "\n".join("  " + l for l in abstract_md.splitlines())

frontmatter = (
    "---\n"
    f"title: \"\\\\textbf{{{title_yaml}}}\"\n"
    "author:\n"
    "  - Faizan Ahmed\n"
    f'date: May 2026\n'
    "abstract: |\n"
    f"{abs_yaml_lines}\n"
    "---\n\n"
)

Path("paper/.draft_article.md").write_text(frontmatter + stripped, encoding="utf-8")
print(f"[build] wrote paper/.draft_article.md  (title='{title[:40]}…')")
PY

# 1. pandoc → tex with article 11pt a4paper + reference's package set
python3 - <<'PY'
import pypandoc
pypandoc.convert_file(
    "paper/.draft_article.md",
    "latex",
    outputfile="paper/draft.tex",
    extra_args=[
        "--standalone",
        "--number-sections",
        "--natbib",
        "--bibliography=paper/refs.bib",
        "-V", "documentclass=article",
        "-V", "classoption=11pt",
        "-V", "classoption=a4paper",
        "-V", "geometry=margin=1in",
        "--include-in-header=paper/article_header.tex",
    ],
)
print("[build] pandoc → paper/draft.tex")
PY

# 2. unicode subs + bibliography in-place + author multi-line block
python3 - <<'PY'
import re
from pathlib import Path
p = Path("paper/draft.tex")
src = p.read_text(encoding="utf-8")
subs = {
    "α": r"\ensuremath{\alpha}",   "β": r"\ensuremath{\beta}",
    "Δ": r"\ensuremath{\Delta}",   "≈": r"\ensuremath{\approx}",
    "∈": r"\ensuremath{\in}",      "‖": r"\ensuremath{\|}",
    "×": r"\ensuremath{\times}",   "→": r"\ensuremath{\rightarrow}",
    "±": r"\ensuremath{\pm}",      "←": r"\ensuremath{\leftarrow}",
    "†": r"\ensuremath{\dagger}",
    "−": r"\ensuremath{-}",        "·": r"\ensuremath{\cdot}",
    "§": r"\S{}",                  "≥": r"\ensuremath{\geq}",
    "≤": r"\ensuremath{\leq}",     "≫": r"\ensuremath{\gg}",
    "⁰": r"\ensuremath{^{0}}", "¹": r"\ensuremath{^{1}}",
    "²": r"\ensuremath{^{2}}", "³": r"\ensuremath{^{3}}",
    "⁴": r"\ensuremath{^{4}}", "⁵": r"\ensuremath{^{5}}",
    "⁶": r"\ensuremath{^{6}}", "⁷": r"\ensuremath{^{7}}",
    "⁸": r"\ensuremath{^{8}}", "⁹": r"\ensuremath{^{9}}",
    "⁻": r"\ensuremath{^{-}}",
}
for ch, latex in subs.items():
    src = src.replace(ch, latex)
# Strip pandoc-injected `hidelinks,` line from \hypersetup{...}. We already
# pass colorlinks=true via article_header.tex's \PassOptionsToPackage; the
# default-template hidelinks would override that and produce uncolored links.
src = re.sub(r"^  hidelinks,\s*\n", "", src, flags=re.M)
src = re.sub(r"\\bibliography\{[^}]*refs(?:\.bib)?\}", r"\\bibliography{refs}", src)
matches = list(re.finditer(r"\\bibliography\{refs\}", src))
if matches:
    src = re.sub(r"\\bibliography\{refs\}\s*", "", src)
    refs_heading_re = re.compile(
        r"\\(?:sub)?section\{(?:\\texorpdfstring\{)?References"
        r"(?:\\newpage)?(?:\}\{References\})?\}\\label\{references\}"
    )
    m = refs_heading_re.search(src)
    if m is None:
        raise RuntimeError("could not locate References heading; bibliography would land at end-of-doc")
    # Replace the (possibly \texorpdfstring{...\newpage}{...}-wrapped) heading
    # with a clean `\section{References}\label{references}` so the inline
    # \newpage that pandoc folded in doesn't push the bibliography to the
    # next page. Then inject \bibliography{refs} right after.
    src = (src[:m.start()]
           + r"\section{References}\label{references}"
           + "\n\n" + r"\bibliography{refs}"
           + src[m.end():])
# Replace single-string \author{Faizan Ahmed} with the multi-line
# Name \\ Affiliation \\ \texttt{email} block (pandoc+YAML mangle inline \\).
src = re.sub(
    r"\\author\{Faizan Ahmed\}",
    r"\\author{Faizan Ahmed \\\\ Headstarter \\\\ "
    r"\\texttt{faizan@theheadstarter.com}}",
    src,
    count=1,
)
p.write_text(src, encoding="utf-8")
print("[build] unicode subs + bibliography moved in-place + multi-line author")
PY

# 3. pdflatex × 2 → bibtex → pdflatex × 2
cd paper
rm -f draft.aux draft.log draft.out draft.toc draft.bbl draft.blg draft.pdf
pdflatex -interaction=nonstopmode -halt-on-error draft.tex >/dev/null
bibtex draft >/dev/null 2>&1 || { echo "bibtex pass failed"; cat draft.blg; exit 1; }
pdflatex -interaction=nonstopmode -halt-on-error draft.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error draft.tex >/dev/null
echo "wrote paper/draft.pdf:"
ls -la draft.pdf
