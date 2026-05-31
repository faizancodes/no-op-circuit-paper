"""Layerwise linear probes for buggy vs fixed (phase 2).

Given a set of per-(task, condition, variant) activation `.pt` files cached by
`modal_app.cache_activations`, train a logistic-regression probe at each
(layer, position) and report cross-validated AUC.

The probe predicts CONDITION (buggy=1, fixed=0). A high AUC at a given layer
× position means the residual stream linearly separates the two trajectories
there — i.e., the model has, by that point, internally encoded "patch
warranted" vs "no patch warranted".

Inputs are loaded from local disk after `modal volume get noop-activations
<run_id>/ <local_dir>`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class ProbeDataset:
    """All activation samples for a single variant.

    features: (N, L, K, D) float32   — resid_post, last K positions
    labels  : (N,) int8              — 1 if buggy, 0 if fixed
    task_ids: list[str] of length N
    """
    features: np.ndarray
    labels: np.ndarray
    task_ids: list[str]
    variant: str
    n_layers: int
    n_positions: int
    hidden: int


def load_run(run_dir: Path, *, variant_filter: str | None = None) -> dict[str, ProbeDataset]:
    """Load every per-prompt .pt under a run dir into per-variant ProbeDatasets.

    Expected layout:
        <run_dir>/manifest.json
        <run_dir>/<task_id>/<condition>__<variant>.pt
                or with the manifest path layout produced by cache_activations.

    We tolerate either flat or nested layouts.
    """
    import torch  # local-only dep

    by_variant: dict[str, dict[str, dict[str, dict]]] = {}
    # by_variant[variant][task_id][condition] = payload dict

    for pt in sorted(run_dir.rglob("*.pt")):
        name = pt.stem  # "<condition>__<variant>"
        if "__" not in name:
            continue
        condition, variant = name.split("__", 1)
        if variant_filter and variant != variant_filter:
            continue
        if condition not in ("buggy", "fixed"):
            continue
        payload = torch.load(pt, map_location="cpu", weights_only=False)
        task_id = payload["task_id"]
        by_variant.setdefault(variant, {}).setdefault(task_id, {})[condition] = payload

    out: dict[str, ProbeDataset] = {}
    for variant, by_task in by_variant.items():
        feats: list[np.ndarray] = []
        labels: list[int] = []
        task_ids: list[str] = []
        for task_id in sorted(by_task):
            sides = by_task[task_id]
            if "buggy" not in sides or "fixed" not in sides:
                continue  # only use complete pairs
            for cond, lab in (("buggy", 1), ("fixed", 0)):
                resid_post = sides[cond]["resid_post"]  # (L, B, K, D)
                if resid_post is None:
                    continue
                # squeeze the batch dim
                arr = resid_post.squeeze(1).to(dtype=torch.float32).numpy()  # (L, K, D)
                feats.append(arr)
                labels.append(lab)
                task_ids.append(task_id)
        if not feats:
            continue
        x = np.stack(feats, axis=0)  # (N, L, K, D)
        y = np.array(labels, dtype=np.int8)
        out[variant] = ProbeDataset(
            features=x,
            labels=y,
            task_ids=task_ids,
            variant=variant,
            n_layers=x.shape[1],
            n_positions=x.shape[2],
            hidden=x.shape[3],
        )
    return out


@dataclass
class ProbeResult:
    variant: str
    layer: int
    position: int        # negative index, e.g. -1 = action position
    auc: float
    n_train: int
    n_test: int


def train_layerwise(
    ds: ProbeDataset,
    *,
    positions: Sequence[int] | None = None,
    n_splits: int = 5,
    random_state: int = 0,
    C: float = 1.0,
) -> list[ProbeResult]:
    """Train a separate logistic-regression probe at every (layer, position).

    Reports stratified-K-fold AUC. With as few as 20-50 paired tasks per
    variant, the labels are perfectly balanced (every task contributes one
    buggy + one fixed), so simple LR with L2 is appropriate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    N, L, K, D = ds.features.shape
    if positions is None:
        # default: every cached position (typically last_k = 32)
        positions = list(range(-K, 0))
    # convert negative indices to absolute
    abs_positions = [p % K for p in positions]

    results: list[ProbeResult] = []
    for layer in range(L):
        for pos_abs, pos_signed in zip(abs_positions, positions):
            X = ds.features[:, layer, pos_abs, :]  # (N, D)
            y = ds.labels
            if len(np.unique(y)) < 2:
                continue
            try:
                skf = StratifiedKFold(n_splits=min(n_splits, N // 2), shuffle=True, random_state=random_state)
                aucs: list[float] = []
                for tr, te in skf.split(X, y):
                    clf = LogisticRegression(C=C, max_iter=2000, solver="liblinear")
                    clf.fit(X[tr], y[tr])
                    scores = clf.decision_function(X[te])
                    aucs.append(float(roc_auc_score(y[te], scores)))
                auc = float(np.mean(aucs))
                n_test = N - (N - N // n_splits)  # rough; not used by callers
            except Exception:
                continue
            results.append(
                ProbeResult(
                    variant=ds.variant,
                    layer=layer,
                    position=pos_signed,
                    auc=auc,
                    n_train=N - n_test,
                    n_test=n_test,
                )
            )
    return results


def fit_steering_vector(
    ds: ProbeDataset,
    *,
    layer: int,
    position: int = -1,
) -> np.ndarray:
    """Mean-difference steering vector: μ_buggy - μ_fixed at (layer, position).

    Add this to a fixed-condition residual to push toward `edit`; subtract from
    a buggy-condition residual to push toward `noop`.
    """
    pos_abs = position % ds.n_positions
    buggy = ds.features[ds.labels == 1, layer, pos_abs, :]
    fixed = ds.features[ds.labels == 0, layer, pos_abs, :]
    return buggy.mean(axis=0) - fixed.mean(axis=0)


def results_to_json(results: list[ProbeResult]) -> str:
    return json.dumps(
        [
            {
                "variant": r.variant,
                "layer": r.layer,
                "position": r.position,
                "auc": r.auc,
                "n_train": r.n_train,
                "n_test": r.n_test,
            }
            for r in results
        ],
        indent=2,
    )
