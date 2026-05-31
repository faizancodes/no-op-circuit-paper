#!/usr/bin/env python3
"""Phase 4: keyword + bag-of-words baselines for the deterministic paraphrase eval.

The paper's paraphrase control replaces the pytest transcript with a fixed NL
paraphrase keyed only on buggy/fixed polarity (schema.paraphrase_test_text;
styles `minimal`, `realistic`). The residual monitor reaches AUC 0.995 there; the
literal-`FAILED` regex collapses to 0.5. This script checks the paper's own
caveat -- that a vocabulary-matched surface baseline solves it -- with a generic
keyword lexicon and a bag-of-words separator. Pure local, no GPU.

Output: results/paraphrase_baselines/paraphrase_baselines.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from no_op_circuit.dataset.schema import paraphrase_test_text

STYLES = ["minimal", "realistic"]
# Generic failure / pass concept lexicon (substring, case-insensitive). Chosen
# to be template-agnostic, not tuned to one style's exact wording.
FAIL = ["absent", "did not", "does not", "differ", "not yet", "mismatch", "fail", "error", "incorrect"]
PASS = ["present", "matched", "aligned", "satisfies", "all test cases", "correct", "passed", "succeed"]


def fail_score(text: str) -> int:
    t = text.lower()
    return sum(t.count(k) for k in FAIL) - sum(t.count(k) for k in PASS)


def bow_vec(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for w in re.findall(r"[a-z]+", text.lower()):
        out[w] = out.get(w, 0) + 1
    return out


def auc_two_class(pos_scores, neg_scores) -> float:
    """AUC = P(score(pos) > score(neg)) with ties = 0.5."""
    n = len(pos_scores) * len(neg_scores)
    if n == 0:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos_scores for q in neg_scores)
    return wins / n


def main():
    out_dir = Path("results/paraphrase_baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    N = 499  # eval size per condition (paper's Qwen paraphrase eval); strings are identical per condition

    result = {"note": "positive class = buggy (failing); N per condition = %d (identical strings)" % N,
              "monitor_auc_realistic_qwen": 0.995, "literal_failed_regex_auc": 0.5,
              "lexicon": {"FAIL": FAIL, "PASS": PASS}, "styles": {}}

    for style in STYLES:
        buggy = paraphrase_test_text("buggy", style)
        fixed = paraphrase_test_text("fixed", style)
        # Keyword baseline: fail_score, buggy should exceed fixed.
        kb, kf = fail_score(buggy), fail_score(fixed)
        kw_auc = auc_two_class([kb] * N, [kf] * N)
        # BoW separator: direction = vec(buggy) - vec(fixed); score = dot(vec, dir).
        vb, vf = bow_vec(buggy), bow_vec(fixed)
        direction = {w: vb.get(w, 0) - vf.get(w, 0) for w in set(vb) | set(vf)}

        def proj(v):
            return sum(v.get(w, 0) * d for w, d in direction.items())

        bow_auc = auc_two_class([proj(vb)] * N, [proj(vf)] * N)
        result["styles"][style] = {
            "buggy_text": buggy.strip(), "fixed_text": fixed.strip(),
            "keyword_fail_score": {"buggy": kb, "fixed": kf}, "keyword_auc": kw_auc,
            "bow_auc": bow_auc,
        }

    # Cross-style: does the SAME generic lexicon separate the other style?
    result["cross_style_keyword_auc"] = {
        s: auc_two_class([fail_score(paraphrase_test_text("buggy", s))] * N,
                         [fail_score(paraphrase_test_text("fixed", s))] * N)
        for s in STYLES
    }
    (out_dir / "paraphrase_baselines.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
