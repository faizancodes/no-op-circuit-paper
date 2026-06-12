"""Server-side monitor projection (Phase 5) — avoids downloading big caches.

`run_monitor_real.py` projects each SWE prompt's residual onto a frozen unit
`v_noop` and computes ROC-AUC. The residual caches live on the Modal Volume and
are large (≈5 GB for 3B, ≈21 GB for 32B), so we read them *server-side* and
return only the small per-task projection scalars; the local entrypoint computes
the AUC. Works for any model size with no big local download.

    modal run -m modal_app.analyze \\
        --cache-run-id cache-qwen25_coder_3b_instruct-20260612T025109Z \\
        --v-noop results/steer-…/v_noop.pt --layer 32 --position -1
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import ACTIVATIONS_DIR, activations_vol, app
from no_op_circuit.config import RESULTS_DIR


@app.function(timeout=60 * 30, volumes={ACTIVATIONS_DIR: activations_vol})
def project_cache(
    cache_run_id: str,
    direction: list[float],
    *,
    layer: int,
    position: int = -1,
    variant: str = "code_tests",
) -> dict[str, Any]:
    """Project every cached (task, condition) residual at (layer, position) onto
    the unit `direction`. Returns {task_id: {cond: {proj, argmax}}} — small."""
    import torch

    activations_vol.reload()
    v = torch.tensor(direction, dtype=torch.float32)
    v = v / v.norm()

    run_dir = Path(ACTIVATIONS_DIR) / cache_run_id
    out: dict[str, dict[str, Any]] = {}
    n = 0
    for pt in sorted(run_dir.rglob(f"*__{variant}.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        rp = payload["resid_pre"]
        if rp is None:
            continue
        K = rp.shape[2]
        pos_abs = position if position >= 0 else K + position
        proj = float(rp[layer, 0, pos_abs, :].float() @ v)
        al = payload["action_logits"]
        argmax = max(al.items(), key=lambda kv: kv[1]["logit"])[0]
        out.setdefault(payload["task_id"], {})[cond] = {"proj": proj, "argmax": argmax}
        n += 1
    print(f"[project_cache] projected {n} prompts from {cache_run_id} at L{layer}/pos{position}")
    return out


@app.function(timeout=60 * 30, volumes={ACTIVATIONS_DIR: activations_vol})
def project_cache_multilayer(
    cache_run_id: str,
    directions: list[list[float]],   # one [hidden] vector per layer
    *,
    position: int = -1,
    variant: str = "code_tests",
) -> dict[str, Any]:
    """Project each cached residual at EVERY layer onto that layer's own
    direction. Returns {task_id: {cond: [proj per layer]}} — small. Used for the
    monitor layer-scan (the causal cell is not necessarily the max-AUC cell)."""
    import torch

    activations_vol.reload()
    D = torch.tensor(directions, dtype=torch.float32)              # (L, hidden)
    D = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
    run_dir = Path(ACTIVATIONS_DIR) / cache_run_id
    out: dict[str, dict[str, Any]] = {}
    n = 0
    for pt in sorted(run_dir.rglob(f"*__{variant}.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        try:
            payload = torch.load(pt, map_location="cpu", weights_only=False)
        except Exception:
            continue
        rp = payload["resid_pre"]
        if rp is None:
            continue
        K = rp.shape[2]
        pos_abs = position if position >= 0 else K + position
        h = rp[:, 0, pos_abs, :].float()                          # (L, hidden)
        proj = (h * D).sum(dim=1)                                 # (L,)
        out.setdefault(payload["task_id"], {})[cond] = proj.tolist()
        n += 1
    print(f"[project_multilayer] projected {n} prompts across {D.shape[0]} layers")
    return out


def _auc(scores_pos, scores_neg) -> float:
    """Rank-based ROC-AUC = P(score_pos > score_neg) (+0.5 ties). numpy only."""
    import numpy as np

    pos = np.asarray(scores_pos, float)
    neg = np.asarray(scores_neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    n_pos = len(pos)
    sum_pos = ranks[: n_pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * len(neg)))


@app.local_entrypoint()
def monitor(
    cache_run_id: str,
    v_noop: str,
    layer: int = 24,
    position: int = -1,
    variant: str = "code_tests",
    out: str = "",
):
    import numpy as np
    import torch

    blob = torch.load(v_noop, map_location="cpu", weights_only=False)
    v = blob["direction"].float()
    if blob.get("layer") is not None and blob["layer"] != layer:
        print(f"[monitor] WARN: v_noop layer={blob['layer']} != --layer {layer}")
    print(f"[monitor] v_noop ||v||={blob.get('norm'):.3f} layer={blob.get('layer')} "
          f"pos={blob.get('position')} N={blob.get('n_pairs')}")

    by_task = project_cache.remote(
        cache_run_id, v.tolist(), layer=layer, position=position, variant=variant
    )
    pairs = [(t, s["buggy"], s["fixed"]) for t, s in by_task.items()
             if "buggy" in s and "fixed" in s]
    print(f"[monitor] paired tasks: {len(pairs)}")

    pb = np.asarray([b["proj"] for _, b, _ in pairs])
    pf = np.asarray([f["proj"] for _, _, f in pairs])
    gap = float(pf.mean() - pb.mean())
    # score = -proj (higher => buggy/failing); positive class = buggy
    auc = _auc((-pb).tolist(), (-pf).tolist())
    print(f"\n=== monitor (frozen v_noop) on {cache_run_id} ===")
    print(f"  proj buggy mean={pb.mean():+.3f}  fixed mean={pf.mean():+.3f}  gap={gap:+.3f}")
    print(f"  ** ROC-AUC = {auc:.4f} **  (N={len(pairs)} pairs)")

    from collections import Counter
    actions = ["view", "grep", "test", "edit", "noop"]
    for lbl in ("buggy", "fixed"):
        c = Counter([(b if lbl == "buggy" else f)["argmax"] for _, b, f in pairs])
        n = sum(c.values()) or 1
        print(f"  argmax {lbl:<5} " + "  ".join(f"{a}:{100*c.get(a,0)/n:.0f}%" for a in actions))

    out_path = Path(out) if out else RESULTS_DIR / "monitor_real" / f"{cache_run_id}_monitor.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "cache_run_id": cache_run_id, "layer": layer, "position": position,
        "roc_auc": auc, "proj_buggy_mean": float(pb.mean()),
        "proj_fixed_mean": float(pf.mean()), "gap": gap, "n_pairs": len(pairs),
    }, indent=2))
    print(f"[monitor] wrote {out_path}")


@app.local_entrypoint()
def layer_scan(
    toy_cache_dir: str,        # local dir of the toy code_tests cache
    swe_cache_run_id: str,     # SWE cache run_id on the Volume
    position: int = -1,
    variant: str = "code_tests",
    causal_layer: int = -1,    # annotate the causal-peak layer in the printout
    out: str = "",
):
    """Per-layer monitor AUC: derive each layer's toy contrast direction, project
    the SWE cache at that layer, compute AUC(layer). Shows whether the causal
    cell is the max-AUC cell (the paper's L12>L24 phenomenon)."""
    from collections import defaultdict
    from pathlib import Path

    import numpy as np
    import torch

    # 1) per-layer toy directions = mean(fixed) - mean(buggy) at (layer, position)
    by_task: dict[str, dict[str, "torch.Tensor"]] = defaultdict(dict)
    for pt in sorted(Path(toy_cache_dir).rglob(f"*__{variant}.pt")):
        cond = pt.stem.split("__", 1)[0]
        if cond not in ("buggy", "fixed"):
            continue
        try:
            payload = torch.load(pt, map_location="cpu", weights_only=False)
        except Exception as exc:
            print(f"[layer_scan] skip unreadable {pt.name}: {exc}")
            continue
        rp = payload["resid_pre"]
        if rp is None:
            continue
        K = rp.shape[2]
        pos_abs = position if position >= 0 else K + position
        by_task[payload["task_id"]][cond] = rp[:, 0, pos_abs, :].float()  # (L, hidden)

    bsum = fsum = None
    n = 0
    for sides in by_task.values():
        if "buggy" in sides and "fixed" in sides:
            bsum = sides["buggy"] if bsum is None else bsum + sides["buggy"]
            fsum = sides["fixed"] if fsum is None else fsum + sides["fixed"]
            n += 1
    if n == 0 or bsum is None or fsum is None:
        raise SystemExit(f"no paired toy tasks in {toy_cache_dir}")
    directions = (fsum / n) - (bsum / n)         # (L, hidden)
    n_layers = directions.shape[0]
    print(f"[layer_scan] toy directions from N={n} pairs · {n_layers} layers")

    # 2) project the SWE cache at every layer (server-side)
    proj = project_cache_multilayer.remote(
        swe_cache_run_id, directions.tolist(), position=position, variant=variant
    )
    pairs = [(t, s["buggy"], s["fixed"]) for t, s in proj.items()
             if "buggy" in s and "fixed" in s]
    print(f"[layer_scan] SWE paired tasks: {len(pairs)}")

    # 3) AUC per layer (score = -proj; positive class = buggy/failing)
    aucs = []
    for L in range(n_layers):
        pb = [-b[L] for _, b, _ in pairs]
        pf = [-f[L] for _, _, f in pairs]
        aucs.append(_auc(pb, pf))
    aucs = np.asarray(aucs)
    best = int(aucs.argmax())
    print(f"\n=== monitor AUC by layer ({swe_cache_run_id}) ===")
    for L in range(n_layers):
        mark = "  <= MAX" if L == best else ("  <= causal" if L == causal_layer else "")
        bar = "#" * int(max(0, (aucs[L] - 0.5)) * 40)
        print(f"  L{L:>2}: AUC={aucs[L]:.4f} {bar}{mark}")
    print(f"\n  MAX-AUC layer: L{best} (rel {best/n_layers:.3f}) AUC={aucs[best]:.4f}")
    if 0 <= causal_layer < n_layers:
        print(f"  causal-peak layer: L{causal_layer} (rel {causal_layer/n_layers:.3f}) AUC={aucs[causal_layer]:.4f}")
        print(f"  => AUC(max) − AUC(causal) = {aucs[best]-aucs[causal_layer]:+.4f} "
              f"(>0 ⇒ discriminability ≠ causal use, as in the paper)")

    out_path = Path(out) if out else RESULTS_DIR / "monitor_real" / f"{swe_cache_run_id}_layerscan.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "swe_cache_run_id": swe_cache_run_id, "n_layers": n_layers,
        "auc_by_layer": aucs.tolist(), "max_layer": best, "max_auc": float(aucs[best]),
        "causal_layer": causal_layer,
        "causal_auc": float(aucs[causal_layer]) if 0 <= causal_layer < n_layers else None,
    }, indent=2))
    print(f"[layer_scan] wrote {out_path}")
