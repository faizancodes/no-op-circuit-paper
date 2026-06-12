#!/usr/bin/env python3
"""Rigor pass on existing data (no new compute): bootstrap CIs on every headline
number, a leave-one-out (held-out) veto threshold, the steering flip-point, and
within-data cuts. Writes results/paper_stats.{json,md}."""

from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

RNG = np.random.default_rng(0)
B = 10000

BEHAV = {
    "3B": "results/cache-qwen25_coder_3b_instruct-20260612T025109Z/manifest.json",
    "7B": "results/cache-qwen25_coder_7b_instruct-20260612T040641Z/manifest.json",
    "14B": "results/cache-qwen25_coder_14b_instruc-20260612T040656Z/manifest.json",
    "32B": "results/cache-qwen25_coder_32b_instruc-20260612T040759Z/manifest.json",
}
PATCH = {
    "3B": ("results/patch-qwen25_coder_3b_instruct-20260612T025131Z", 36),
    "7B": ("results/patch-qwen25_coder_7b_instruct-20260612T052705Z", 28),
    "14B": ("results/patch-qwen25_coder_14b_instruc-20260612T052719Z", 48),
    "32B": ("results/patch-qwen25_coder_32b_instruc-20260612T052732Z", 64),
}
STEER_3B = "results/steer-qwen25_coder_3b_instruct-20260612T063444Z/manifest.json"
LOOPS = sorted(glob.glob("results/agentloop/*_loops*.json"))

out: dict = {}
md: list[str] = ["# Statistical rigor pass (bootstrap CIs, held-out veto, flip-point)\n",
                 "All from existing data; bootstrap B=10,000, seed 0; CIs are 95% percentile.\n"]


def ci(values, stat, b=B):
    v = np.asarray(values, float)
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"))
    boots = np.array([stat(v[RNG.integers(0, n, n)]) for _ in range(b)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def auc_ci(pos, neg, b=2000):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    boots = [auc(pos[RNG.integers(0, len(pos), len(pos))], neg[RNG.integers(0, len(neg), len(neg))])
             for _ in range(b)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------- F1: noop-on-fixed rate per size, with CI ----------
md.append("## F1 — noop (do-nothing) rate on passing prompts\n")
md.append("| size | noop on fixed | 95% CI |\n|---|---|---|")
out["F1"] = {"1.5B": {"rate": 0.0, "note": "paper baseline"}}
md.append("| 1.5B | 0.0% | (paper) |")
for sz, path in BEHAV.items():
    m = json.load(open(path))
    by = defaultdict(dict)
    for e in m["entries"]:
        by[e["task_id"]][e["condition"]] = e["action_argmax"]
    fixed_noop = np.array([1.0 if s.get("fixed") == "noop" else 0.0
                           for s in by.values() if "fixed" in s])
    lo, hi = ci(fixed_noop, np.mean)
    out["F1"][sz] = {"rate": float(fixed_noop.mean()), "ci": [lo, hi], "n": int(len(fixed_noop))}
    md.append(f"| {sz} | {100*fixed_noop.mean():.1f}% | [{100*lo:.1f}, {100*hi:.1f}] |")

# ---------- F3: patch peak per size, effect CI + Wilcoxon, peak-layer stability ----------
md.append("\n## F3 — causal patch peak (bidirectional-min), effect CI + sign test\n")
md.append("| size | peak L/pos | rel-depth | F→B mean [95% CI] | Wilcoxon p(>0) | peak-layer stability |\n|---|---|---|---|---|---|")
out["F3"] = {"1.5B": {"peak_layer": 24, "rel_depth": 0.857, "note": "paper"}}
import torch  # noqa: E402
from scipy import stats as sps  # noqa: E402
for sz, (pdir, nlayers) in PATCH.items():
    files = sorted(glob.glob(f"{pdir}/**/*__patch.pt", recursive=True))
    p0 = torch.load(files[0], map_location="cpu", weights_only=False)
    L, P = p0["layer_indices"], p0["position_offsets"]
    f2b = np.zeros((len(files), len(L), len(P)))
    b2f = np.zeros((len(files), len(L), len(P)))
    for i, f in enumerate(files):
        p = torch.load(f, map_location="cpu", weights_only=False)
        cb, cf = p["clean_buggy_margin"], p["clean_fixed_margin"]
        f2b[i] = cb - np.array(p["patched_margins_f2b"])
        b2f[i] = np.array(p["patched_margins_b2f"]) - cf
    bidir = np.minimum(f2b.mean(0), b2f.mean(0))
    li, pi = np.unravel_index(np.argmax(bidir), bidir.shape)
    peak_f2b = f2b[:, li, pi]
    lo, hi = ci(peak_f2b, np.mean)
    wp = sps.wilcoxon(peak_f2b, alternative="greater").pvalue
    # peak-layer stability: bootstrap the argmax layer
    peak_layers = []
    for _ in range(2000):
        idx = RNG.integers(0, len(files), len(files))
        bd = np.minimum(f2b[idx].mean(0), b2f[idx].mean(0))
        peak_layers.append(L[np.unravel_index(np.argmax(bd), bd.shape)[0]])
    pl = np.array(peak_layers)
    frac_mode = float((pl == L[li]).mean())
    out["F3"][sz] = {"peak_layer": int(L[li]), "pos": int(P[pi]), "rel_depth": L[li] / nlayers,
                     "f2b_mean": float(peak_f2b.mean()), "ci": [lo, hi], "wilcoxon_p": float(wp),
                     "peak_layer_boot_mode_frac": frac_mode}
    md.append(f"| {sz} | L{L[li]}/{P[pi]} | {L[li]/nlayers:.3f} | "
              f"{peak_f2b.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {wp:.1e} | L{L[li]} in {100*frac_mode:.0f}% of boots |")

# ---------- Steering flip-point (3B, fixed) ----------
md.append("\n## Steering flip-point (3B, passing prompts)\n")
sm = json.load(open(STEER_3B))
rows = sm["rows"]
alphas = sorted(set(r["alpha"] for r in rows))
noop_rate, margin = {}, {}
for a in alphas:
    fr = [r for r in rows if r["condition"] == "fixed" and r["alpha"] == a]
    noop_rate[a] = np.mean([r["argmax_action"] == "noop" for r in fr])
    margin[a] = np.mean([r["edit_minus_noop"] for r in fr])
flip_alpha = next((a for a in alphas if noop_rate[a] >= 0.5), None)
# linear-interp zero-crossing of mean margin
zc = None
for a1, a2 in zip(alphas, alphas[1:]):
    if margin[a1] > 0 >= margin[a2]:
        zc = a1 + (a2 - a1) * margin[a1] / (margin[a1] - margin[a2])
        break
out["steering"] = {"noop_majority_alpha": flip_alpha, "margin_zero_crossing_alpha": zc,
                   "noop_rate_by_alpha": {str(a): float(noop_rate[a]) for a in alphas}}
md.append(f"- noop becomes majority on passing prompts at **α = {flip_alpha:+.1f}** "
          f"(0%→{100*max(noop_rate.values()):.0f}%); mean edit−noop margin crosses 0 at **α ≈ {zc:.2f}**.")
md.append(f"- On buggy prompts noop stays low until higher α (evidence-tracking asymmetry; see KEYSTONE).")

# ---------- Binary {edit,noop} noop-on-fixed per size, with CI ----------
md.append("\n## Binary {edit,noop} — noop on passing, with CI\n")
md.append("| size | noop on fixed | 95% CI |\n|---|---|---|")
out["binary"] = {}
for sz in ["1.5B", "3B", "7B", "14B", "32B"]:
    f = f"results/action_order_control/qwen_{sz}_binary_scores.json"
    d = json.load(open(f))
    by = defaultdict(list)
    for r in d["rows"]:
        if r["condition"] == "fixed":
            by[r["task_id"]].append(1.0 if r["argmax"] == "noop" else 0.0)
    pertask = np.array([np.mean(v) for v in by.values()])
    lo, hi = ci(pertask, np.mean)
    out["binary"][sz] = {"rate": float(pertask.mean()), "ci": [lo, hi]}
    md.append(f"| {sz} | {100*pertask.mean():.1f}% | [{100*lo:.1f}, {100*hi:.1f}] |")

# ---------- Agent loop: over-editing CI, held-out veto, within-data cuts ----------
md.append("\n## Agent loop — over-editing CIs, held-out veto, evidence-gathering cut\n")
out["loop"] = {}


def heldout_veto(fe, be):
    """Leave-one-out Youden threshold; held-out blocked(over-edits)/preserved(correct)."""
    fe, be = np.asarray(fe, float), np.asarray(be, float)
    allp = np.concatenate([fe, be])
    lab = np.concatenate([np.ones(len(fe)), np.zeros(len(be))])  # 1 = over-edit (block)
    blocked, preserved = [], []
    for i in range(len(allp)):
        mask = np.arange(len(allp)) != i
        tp, tl = allp[mask], lab[mask]
        fet, bet = tp[tl == 1], tp[tl == 0]
        cands = np.unique(tp)
        best = (-2.0, cands[0])
        for thr in cands:
            b_ = (fet > thr).mean() if len(fet) else 0
            p_ = (bet <= thr).mean() if len(bet) else 0
            if b_ + p_ - 1 > best[0]:
                best = (b_ + p_ - 1, thr)
        thr = best[1]
        if lab[i] == 1:
            blocked.append(float(allp[i] > thr))
        else:
            preserved.append(float(allp[i] <= thr))
    return np.array(blocked), np.array(preserved)


for f in LOOPS:
    d = json.load(open(f))
    model = d["model"].split("/")[-1]
    if f.endswith("_ev_stop.json"):
        mode = "ev_stop"
    elif f.endswith("_ev.json"):
        mode = "ev"
    else:
        mode = "issue"
    R = d["results"]
    fixed = [r for r in R if r["condition"] == "fixed"]
    buggy = [r for r in R if r["condition"] == "buggy"]
    oe = np.array([1.0 if r["terminal_action"] == "edit" else 0.0 for r in fixed])
    rec = np.array([1.0 if r["terminal_action"] == "edit" else 0.0 for r in buggy])
    lo, hi = ci(oe, np.mean)
    key = f"{model}/{mode}"
    out["loop"][key] = {"over_edit": float(oe.mean()), "over_edit_ci": [lo, hi],
                        "recall": float(rec.mean())}
    md.append(f"\n### {key}")
    md.append(f"- over-editing (passing→edit): **{100*oe.mean():.1f}%** [{100*lo:.1f}, {100*hi:.1f}]; "
              f"correct-edit recall {100*rec.mean():.1f}%")
    if mode == "issue":
        # within-data: tested vs untested over-editing
        t = [r for r in fixed if r["tested"]]
        u = [r for r in fixed if not r["tested"]]
        te = np.mean([r["terminal_action"] == "edit" for r in t]) if t else float("nan")
        ue = np.mean([r["terminal_action"] == "edit" for r in u]) if u else float("nan")
        out["loop"][key].update(tested_over_edit=float(te), untested_over_edit=float(ue),
                                 n_tested=len(t), n_untested=len(u))
        md.append(f"- evidence-gathering cut: over-editing among **tested** loops "
                  f"{100*te:.0f}% (n={len(t)}) vs **untested** {100*ue:.0f}% (n={len(u)})")
    if mode == "ev":
        fe = [r["decision_proj"] for r in fixed if r["terminal_action"] == "edit" and r["decision_proj"] is not None]
        be = [r["decision_proj"] for r in buggy if r["terminal_action"] == "edit" and r["decision_proj"] is not None]
        a = auc(fe, be)
        a_lo, a_hi = auc_ci(fe, be)
        blk, pre = heldout_veto(fe, be)
        blo, bhi = ci(blk, np.mean)
        plo, phi = ci(pre, np.mean)
        oe_after = oe.mean() * (1 - blk.mean())
        out["loop"][key].update(veto_auc=a, blocked=float(blk.mean()), blocked_ci=[blo, bhi],
                                preserved=float(pre.mean()), preserved_ci=[plo, phi],
                                over_edit_after=float(oe_after))
        md.append(f"- monitor-veto AUC(passing-edit vs buggy-edit) = **{a:.3f}** [{a_lo:.3f}, {a_hi:.3f}]")
        md.append(f"- **held-out (LOO) veto**: over-edits BLOCKED **{100*blk.mean():.0f}%** "
                  f"[{100*blo:.0f},{100*bhi:.0f}]; correct-edits PRESERVED **{100*pre.mean():.0f}%** "
                  f"[{100*plo:.0f},{100*phi:.0f}]")
        md.append(f"- => over-editing **{100*oe.mean():.0f}% → {100*oe_after:.0f}%** (held-out threshold)")

# ---------- consolidated ladder: over-editing & veto vs size ----------
md.append("\n## Ladder — over-editing & veto vs size\n")
md.append("| size | over-edit (from-issue) | over-edit (evidence) | veto→ (held-out) | recall (evidence) | explicit-stop: over-edit / recall |")
md.append("|---|---|---|---|---|---|")


def _k(sz, mode):
    return out["loop"].get(f"Qwen2.5-Coder-{sz}-Instruct/{mode}")


for sz in ["1.5B", "3B", "7B", "14B", "32B"]:
    iss, ev, st = _k(sz, "issue"), _k(sz, "ev"), _k(sz, "ev_stop")
    oi = f"{100*iss['over_edit']:.0f}%" if iss else "—"
    oe = f"{100*ev['over_edit']:.0f}%" if ev else "—"
    va = f"{100*ev['over_edit_after']:.0f}%" if ev and "over_edit_after" in ev else "—"
    rc = f"{100*ev['recall']:.0f}%" if ev else "—"
    sp = f"{100*st['over_edit']:.0f}% / {100*st['recall']:.0f}%" if st else "—"
    md.append(f"| {sz} | {oi} | {oe} | {va} | {rc} | {sp} |")
out["ladder_summary"] = {sz: {"issue": _k(sz, "issue"), "ev": _k(sz, "ev"), "ev_stop": _k(sz, "ev_stop")}
                         for sz in ["1.5B", "3B", "7B", "14B", "32B"]}

json.dump(out, open("results/paper_stats.json", "w"), indent=2)
open("results/paper_stats.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
print("\n[wrote results/paper_stats.json + results/paper_stats.md]")
