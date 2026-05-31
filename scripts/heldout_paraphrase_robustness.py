#!/usr/bin/env python
"""Held-out paraphrase robustness (Qwen, exploratory) — no-leakage experiment.

Tests whether Qwen's frozen L24/pos -1 u_tx generalizes beyond one deterministic
paraphrase, vs text baselines fit ONLY on train-template vocabulary.

No-leakage protocol (enforced in code):
  - u_tx is the frozen toy-trained Qwen direction (results/steer-.../v_noop.pt);
    NO layer/cell/threshold/direction fitting on held-out templates.
  - keyword regex is built ONLY from TRAIN-template discriminative tokens.
  - BoW logistic is fit ONLY on train-template transcripts, evaluated on held-out.
  - literal-FAILED regex is the trivial pytest-token baseline (chance on paraphrases).

Positive class = failing-transcript / buggy condition (score = -projection,
higher = failing). Two train templates + two held-out templates; held-out
discriminative vocab is disjoint from train (verified by the printed overlap report).

RUN:
  python scripts/heldout_paraphrase_robustness.py --self-test     # CPU, disjointness + baseline sanity
  python scripts/heldout_paraphrase_robustness.py --run --tasks real   # Modal Qwen scoring + baselines
Outputs (only on --run):
  results/heldout_paraphrase_robustness/qwen_summary.json
  results/heldout_paraphrase_robustness/qwen_scores.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results/heldout_paraphrase_robustness"
U_TX_ARTIFACT = REPO / "results/steer-20260516T021522Z/v_noop.pt"
LAYER = 24
SEED = 0

# Minimal-paraphrase strings rendered by the `code_tests_paraphrased_minimal`
# variant; we string-replace these with our templates (no schema.py change).
_MINIMAL = {
    "buggy": "The expected behavior was absent in the output.",
    "fixed": "The expected behavior was present in the output.",
}

# --- Template banks (condition -> list of templates). -------------------------
# TRAIN templates: text baselines (keyword regex, BoW) may be fit on these.
# NOTE: templates avoid sharing ANY polarity-carrying token across families,
# including function words like "no"/"not" (a shared negation cue leaks polarity
# to a train-fit BoW even when content vocab is disjoint). The printed
# overlap report + the --self-test BoW(train->held-out) check guard this.
TRAIN_TEMPLATES = {
    "buggy": [
        "The check result was wrong. The observed answer diverged from the specification. The implementation still breaks the required behavior.",
        "The trial exposed incorrect behavior. The produced answer departed from the stated requirement. The change is still needed.",
    ],
    "fixed": [
        "The check result was right. The observed answer matched the specification. The implementation now fulfills the required behavior.",
        "The trial confirmed correct behavior. The produced answer followed the stated requirement. The change is unneeded.",
    ],
}
# HELD-OUT templates: disjoint discriminative vocab (incl. no shared negation).
# NO baseline fitting here.
HELDOUT_TEMPLATES = {
    "buggy": [
        "Validation reported a violation. The produced value contradicted the contract. The defect remains open.",
        "Review surfaced an unresolved discrepancy. The run conflicted with the intended contract. Repair remains pending.",
    ],
    "fixed": [
        "Validation cleared the contract. The produced value honored the contract. The defect is closed.",
        "Review affirmed consistent behavior. The run respected the intended contract. Repair is complete.",
    ],
}

_STOP = {
    "the", "a", "an", "is", "was", "were", "be", "been", "to", "from", "of",
    "in", "on", "with", "and", "or", "no", "longer", "still", "now", "found",
    "result", "answer", "behavior", "value", "implementation", "change",
    "observed", "produced", "run", "trial", "check", "review", "validation",
    "detected", "exposed", "confirmed", "specification", "requirement",
    "required", "stated", "intended", "contract", "defect", "repair", "an",
}


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z]+", s.lower())


def _discriminative(templates) -> set[str]:
    words = set()
    for sents in templates.values():
        for s in sents:
            words |= set(_tokens(s))
    return words - _STOP


def overlap_report() -> dict:
    train = _discriminative(TRAIN_TEMPLATES)
    held = _discriminative(HELDOUT_TEMPLATES)
    ov = sorted(train & held)
    return {
        "train_discriminative": sorted(train),
        "heldout_discriminative": sorted(held),
        "overlap": ov,
        "disjoint": len(ov) == 0,
    }


def template_for(condition: str, family: str, idx: int) -> str:
    bank = TRAIN_TEMPLATES if family == "train" else HELDOUT_TEMPLATES
    sents = bank[condition]
    return sents[idx % len(sents)]


# --- metrics (sklearn) --------------------------------------------------------
def auc_ap(scores, labels):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(set(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


# --- text baselines (no leakage) ----------------------------------------------
def literal_failed_scores(texts):
    return [1.0 if re.search(r"\bFAILED\b|\bfailed\b", t) else 0.0 for t in texts]


def keyword_scores(texts):
    """Train-vocab keyword baseline. Pass-words minus fail-words; sign gives
    failing-vs-passing. Built ONLY from TRAIN discriminative tokens that occur
    in exactly one polarity (so it is an honest train-derived keyword list)."""
    fail_words = _discriminative({"x": TRAIN_TEMPLATES["buggy"]}) - _discriminative({"x": TRAIN_TEMPLATES["fixed"]})
    pass_words = _discriminative({"x": TRAIN_TEMPLATES["fixed"]}) - _discriminative({"x": TRAIN_TEMPLATES["buggy"]})
    out = []
    for t in texts:
        toks = set(_tokens(t))
        out.append(float(len(toks & fail_words) - len(toks & pass_words)))  # higher => failing
    return out, sorted(fail_words), sorted(pass_words)


def bow_heldout_scores(train_texts, train_labels, held_texts):
    """BoW logistic fit ONLY on train-template transcripts, scored on held-out.
    Returns P(failing) for held-out texts (positive class = failing = label 1)."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = CountVectorizer(lowercase=True, token_pattern=r"[a-zA-Z]+")
    Xtr = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(Xtr, train_labels)
    Xte = vec.transform(held_texts)
    pos_idx = list(clf.classes_).index(1)
    return [float(p[pos_idx]) for p in clf.predict_proba(Xte)]


# --- self-test ----------------------------------------------------------------
def self_test() -> None:
    rep = overlap_report()
    print("=== template vocabulary-overlap report ===")
    print("train discriminative :", rep["train_discriminative"])
    print("held-out discriminative:", rep["heldout_discriminative"])
    print("overlap:", rep["overlap"], "| disjoint:", rep["disjoint"])
    assert rep["disjoint"], "train/held-out discriminative vocab overlap!"
    # keyword baseline: high on train, chance on held-out (disjoint vocab)
    def fam_eval(fam):
        texts, labels = [], []
        for idx in range(40):
            for cond, lab in (("buggy", 1), ("fixed", 0)):
                texts.append(template_for(cond, fam, idx)); labels.append(lab)
        ks, _, _ = keyword_scores(texts)
        return auc_ap(ks, labels)[0]
    print(f"\nkeyword(train-vocab) AUC: train={fam_eval('train'):.3f} (expect ~1.0), "
          f"held-out={fam_eval('heldout'):.3f} (expect ~0.5)")
    # BoW fit on train, eval held-out
    tr_txt, tr_lab = [], []
    for idx in range(40):
        for cond, lab in (("buggy", 1), ("fixed", 0)):
            tr_txt.append(template_for(cond, "train", idx)); tr_lab.append(lab)
    he_txt, he_lab = [], []
    for idx in range(40):
        for cond, lab in (("buggy", 1), ("fixed", 0)):
            he_txt.append(template_for(cond, "heldout", idx)); he_lab.append(lab)
    bow = bow_heldout_scores(tr_txt, tr_lab, he_txt)
    import statistics
    spread = max(bow) - min(bow)
    print(f"BoW(train->held-out): prediction spread = {spread:.4f} "
          f"(near-zero => no transferable features; AUC is degenerate/ill-defined "
          f"when predictions are near-constant). std={statistics.pstdev(bow):.4f}")
    print("\nSELF-TEST OK (no GPU; no fabricated residual numbers).")


# --- full run -----------------------------------------------------------------
def _render_jobs(family: str):
    """Build per-(task,condition) jobs for a template family + return transcript texts."""
    from no_op_circuit.agent.prompt import build_prompt
    from no_op_circuit.config import DATA_DIR
    from no_op_circuit.dataset import VARIANTS, iter_tasks

    var = VARIANTS["code_tests_paraphrased_minimal"]
    tasks = list(iter_tasks(tasks_dir=DATA_DIR / "real_tasks"))
    jobs, meta = [], []
    for ti, task in enumerate(tasks):
        for cond in ("buggy", "fixed"):
            idx = ti % 2  # round-robin over the 2 templates per family
            tmpl = template_for(cond, family, idx)
            pb = build_prompt(task, cond, var)
            # swap the minimal paraphrase string for our template in the user msg
            replaced = False
            for m in pb.messages:
                if _MINIMAL[cond] in m["content"]:
                    m["content"] = m["content"].replace(_MINIMAL[cond], tmpl)
                    replaced = True
            if not replaced:
                raise RuntimeError(f"minimal paraphrase not found for {task.task_id}/{cond}")
            jobs.append({
                "task_id": task.task_id, "condition": cond,
                "variant_name": f"heldout_{family}_t{idx}",
                "messages": pb.messages, "action_suffix": pb.action_suffix,
            })
            meta.append({"task_id": task.task_id, "condition": cond,
                         "family": family, "template_idx": idx, "transcript": tmpl,
                         "label_failing": 1 if cond == "buggy" else 0})
    return jobs, meta


def run_full(tasks: str, model: str, site_layer: int = LAYER,
             direction_path: str | None = None, output_prefix: str = "qwen",
             gpu: str = "") -> None:
    import sys
    sys.path.insert(0, str(REPO))  # make the top-level modal_app package importable
    import torch
    from modal_app.common import app
    from modal_app import noisy_monitor as nm

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = overlap_report()
    assert rep["disjoint"], f"templates not disjoint: {rep['overlap']}"
    dpath = Path(direction_path) if direction_path else U_TX_ARTIFACT
    blob = torch.load(dpath, map_location="cpu", weights_only=False)
    assert int(blob["layer"]) == site_layer, (blob["layer"], site_layer)
    u = blob["direction"].tolist()
    # A100 scorer for 7B models (CodeGemma) if requested; A10G default otherwise.
    scorer = getattr(nm, "score_monitor_a100", nm.score_monitor) if gpu.lower() == "a100" else nm.score_monitor

    families = ["train", "heldout"]
    rendered = {fam: _render_jobs(fam) for fam in families}
    all_meta, all_scores = {}, {}
    with app.run():
        for fam in families:
            jobs, meta = rendered[fam]
            print(f"[heldout] scoring {len(jobs)} {fam} prompts on Modal "
                  f"({model}, L{site_layer}/pos -1, gpu={gpu or 'A10G'})...")
            rows = scorer.remote(jobs, model_name=model, layer=site_layer, u_tx=u)
            by_key = {(r["task_id"], r["condition"]): r["score"] for r in rows}
            for m in meta:
                m["residual_score"] = by_key[(m["task_id"], m["condition"])]
            all_meta[fam] = meta
            all_scores[fam] = rows

    # ---- residual monitor AUC/AP (positive = failing/buggy) ----
    def resid_metrics(meta):
        s = [m["residual_score"] for m in meta]
        y = [m["label_failing"] for m in meta]
        return auc_ap(s, y)

    held = all_meta["heldout"]
    train = all_meta["train"]
    summary = {
        "model": model, "site": f"L{site_layer}/pos -1",
        "direction_artifact": str(dpath.relative_to(REPO)) if dpath.is_relative_to(REPO) else str(dpath),
        "seed": SEED,
        "templates": {"train": TRAIN_TEMPLATES, "heldout": HELDOUT_TEMPLATES},
        "vocab_overlap_report": rep,
        "n_per_condition": {fam: {"buggy": sum(1 for m in all_meta[fam] if m["condition"] == "buggy"),
                                  "fixed": sum(1 for m in all_meta[fam] if m["condition"] == "fixed")}
                            for fam in families},
        "commands": ["python scripts/heldout_paraphrase_robustness.py --run --tasks real"],
    }
    # residual: train-template and held-out, pooled + per held-out template
    summary["residual_auc_ap"] = {
        "train_pooled": resid_metrics(train),
        "heldout_pooled": resid_metrics(held),
    }
    for idx in (0, 1):
        sub = [m for m in held if m["template_idx"] == idx]
        summary["residual_auc_ap"][f"heldout_template_{idx}"] = resid_metrics(sub)

    # ---- text baselines on held-out (no leakage) ----
    held_texts = [m["transcript"] for m in held]
    held_y = [m["label_failing"] for m in held]
    train_texts = [m["transcript"] for m in train]
    train_y = [m["label_failing"] for m in train]

    lit = literal_failed_scores(held_texts)
    kw, fail_w, pass_w = keyword_scores(held_texts)
    bow = bow_heldout_scores(train_texts, train_y, held_texts)
    summary["baselines_heldout_auc_ap"] = {
        "literal_FAILED_regex": auc_ap(lit, held_y),
        "train_vocab_keyword": auc_ap(kw, held_y),
        "bow_train_to_heldout": auc_ap(bow, held_y),
    }
    summary["keyword_lists"] = {"fail_words": fail_w, "pass_words": pass_w}
    summary["baseline_fit_on_heldout"] = False  # explicit: no leakage

    (OUT_DIR / f"{output_prefix}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (OUT_DIR / f"{output_prefix}_scores.json").write_text(json.dumps(
        {"train": all_meta["train"], "heldout": all_meta["heldout"]}, indent=2, default=str))

    print(f"\n=== HELD-OUT PARAPHRASE ROBUSTNESS ({output_prefix}, disjoint-vocab) ===")
    print(f"residual train-pooled  AUC/AP: {summary['residual_auc_ap']['train_pooled']}")
    print(f"residual held-out pooled AUC/AP: {summary['residual_auc_ap']['heldout_pooled']}")
    for idx in (0, 1):
        print(f"  residual held-out template {idx}: {summary['residual_auc_ap'][f'heldout_template_{idx}']}")
    print(f"baseline literal-FAILED held-out: {summary['baselines_heldout_auc_ap']['literal_FAILED_regex']}")
    print(f"baseline train-vocab keyword held-out: {summary['baselines_heldout_auc_ap']['train_vocab_keyword']}")
    print(f"baseline BoW(train->held-out): {summary['baselines_heldout_auc_ap']['bow_train_to_heldout']}")
    print(f"\nwrote {OUT_DIR}/{output_prefix}_summary.json + {output_prefix}_scores.json")


def reanalyze() -> None:
    """CPU-only: recompute the full per-template metric set from the cached
    qwen_scores.json (no new GPU forwards) and rewrite qwen_summary.json with
    per-template baseline AUCs, BoW prediction spread, and score gaps."""
    import statistics
    scores = json.loads((OUT_DIR / "qwen_scores.json").read_text())
    train, held = scores["train"], scores["heldout"]
    rep = overlap_report()

    def resid_metrics(rows):
        return auc_ap([m["residual_score"] for m in rows], [m["label_failing"] for m in rows])

    def score_gap(rows):
        fail = [m["residual_score"] for m in rows if m["label_failing"] == 1]
        pas = [m["residual_score"] for m in rows if m["label_failing"] == 0]
        return float(statistics.mean(fail) - statistics.mean(pas))

    held_t = {i: [m for m in held if m["template_idx"] == i] for i in (0, 1)}

    # text baselines: fit on ALL train transcripts, evaluate per held-out template + pooled
    train_txt = [m["transcript"] for m in train]
    train_y = [m["label_failing"] for m in train]

    def baselines_for(rows):
        txt = [m["transcript"] for m in rows]
        y = [m["label_failing"] for m in rows]
        lit = auc_ap(literal_failed_scores(txt), y)
        kw, _, _ = keyword_scores(txt)
        kwm = auc_ap(kw, y)
        bow = bow_heldout_scores(train_txt, train_y, txt)
        bowm = auc_ap(bow, y)
        spread = (max(bow) - min(bow)) if bow else float("nan")
        std = statistics.pstdev(bow) if bow else float("nan")
        return {"literal_FAILED": lit, "train_vocab_keyword": kwm,
                "bow_train_to_heldout": bowm, "bow_pred_spread": spread, "bow_pred_std": std}

    summary = json.loads((OUT_DIR / "qwen_summary.json").read_text())
    summary["reanalysis"] = {
        "vocab_overlap_report": rep,
        "preprocessing": "lowercase; [a-z]+ tokenization; stopword/structural-word list stripped (see scripts/heldout_paraphrase_robustness.py _STOP)",
        "residual": {
            "train_pooled": resid_metrics(train),
            "heldout_pooled": resid_metrics(held),
            "heldout_A": resid_metrics(held_t[0]),
            "heldout_B": resid_metrics(held_t[1]),
            "heldout_A_score_gap": score_gap(held_t[0]),
            "heldout_B_score_gap": score_gap(held_t[1]),
            "heldout_pooled_score_gap": score_gap(held),
        },
        "baselines_pooled": baselines_for(held),
        "baselines_heldout_A": baselines_for(held_t[0]),
        "baselines_heldout_B": baselines_for(held_t[1]),
    }
    (OUT_DIR / "qwen_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    r = summary["reanalysis"]
    print("=== G.17 reanalysis (CPU, from cached scores) ===")
    print(f"vocab overlap: {rep['overlap']} | disjoint(content): {rep['disjoint']}")
    print(f"residual heldout: A AUC={r['residual']['heldout_A'][0]:.3f}  B AUC={r['residual']['heldout_B'][0]:.3f}  "
          f"pooled AUC={r['residual']['heldout_pooled'][0]:.3f} AP={r['residual']['heldout_pooled'][1]:.3f}")
    print(f"residual train pooled AUC={r['residual']['train_pooled'][0]:.3f}")
    for name in ("literal_FAILED", "train_vocab_keyword", "bow_train_to_heldout"):
        pa = r["baselines_pooled"][name]; a = r["baselines_heldout_A"][name]; b = r["baselines_heldout_B"][name]
        print(f"  {name:<22} A={a[0]:.3f} B={b[0]:.3f} pooled AUC={pa[0]:.3f} AP={pa[1]:.3f}")
    print(f"BoW pooled pred spread={r['baselines_pooled']['bow_pred_spread']:.4f} std={r['baselines_pooled']['bow_pred_std']:.4f}")
    print(f"\nrewrote {OUT_DIR}/qwen_summary.json (added 'reanalysis')")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--reanalyze", action="store_true",
                    help="CPU-only: recompute per-template metrics from cached qwen_scores.json")
    ap.add_argument("--tasks", default="real")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--site-layer", type=int, default=LAYER)
    ap.add_argument("--direction-path", default=None,
                    help="frozen toy-trained direction artifact (default: Qwen v_noop)")
    ap.add_argument("--output-prefix", default="qwen")
    ap.add_argument("--gpu", default="", help="'a100' to use the A100 scorer for 7B models")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    elif args.run:
        run_full(args.tasks, args.model, args.site_layer, args.direction_path,
                 args.output_prefix, args.gpu)
    elif args.reanalyze:
        reanalyze()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
