# Step 3 — Qwen2.5-Coder-3B calibration + first scaling data point

Run via `modal run --detach … --spawn` (laptop-independent). Both jobs ran fully on Modal.
- cache run: `cache-qwen25_coder_3b_instruct-20260612T025109Z` (Phase 1, 499 SWE pairs, code_tests)
- patch run: `patch-qwen25_coder_3b_instruct-20260612T025131Z` (Phase 3, 49 toys, bidirectional, layer_step 2)

## F1 — behavioral (499 SWE prompts, first-token argmax)
| metric | Qwen-1.5B (paper) | **Qwen-3B (this run)** |
|---|---|---|
| noop argmax rate (overall) | 0% | **5.5% (55/998)** |
| noop on buggy/failing | 0% | **0% (0/499)** |
| noop on fixed/passing | 0% | **11.0% (55/499)** |
| dominant hedge | grep ~90% | **edit (912/998)** |
| behavioral buggy−fixed margin gap | +0.66 | **+1.97** |

**Reading:** the bigger model starts choosing *do-nothing* — and only when the tests pass
(0% on failing prompts). The represented-evidence-vs-action dissociation (F7) begins to
**close with scale**, on the very first larger model. Caveats: single model, static
single-turn prompt, first-token argmax (not a live agent loop); the hedge flipped grep→edit,
so part of the noop competitiveness is the different 3B action prior — the action-menu /
letter-only controls (Phase 6) are needed to separate prior from evidence.

## F3 — causal localization (49 toys, bidirectional patch grid, coarse layer_step 2)
- **Peak: L32 / pos −1, rel-depth 0.889** — inside the pre-registered band (L30–33 / 0.82–0.93).
  (1.5B peak was L24 / rel-depth 0.857.)
- Layer profile at pos −1: flat ≤L24, sharp monotonic ramp L26→L34 (+0.39→+1.00).
- Bidirectional at peak: F→B +1.00, B→F +1.01; recovers ~80% of the +1.25 toy gap
  (coarse grid undershoots; a fine `layer_step 1` sweep around L30–34 should sharpen it).
- Same mechanism signature as the paper's 1.5B; **relative-depth law holds**.

## Timing → cost recalibration (anchored on real A10G numbers)
| phase | work | wall (A10G) | cost |
|---|---|---|---|
| Phase 1 cache | 998 SWE prompts | ~226 s | ~$0.07 |
| Phase 3 coarse patch grid | 49 toys × 18 layers × 8 pos × 2 dir (~14.3k fwd) | ~1740 s | ~$0.53 |

Per-forward: SWE cache ~0.2 s (long prompts + all-layer resid cache), toy patch ~0.12 s.
The patch grid is the long pole. These anchors keep the full-study estimate in the
**~$35–50** range (broadly the earlier $48), patch grids dominant; coarse-one-way + fine-band
+ `--max-suffix 4` would trim ~30%. Spend so far (smoke + 3B): ≈ $1.3.

## F4 — rank-1 steering (49 toys, v_noop at L32/pos −1)
Dose-response is smooth and monotonic; adding the fixed-direction drives edit−noop down:
| α | buggy | fixed |
|---|---|---|
| −3 | +8.44 | +6.25 |
| 0 | +3.66 | +2.40 |
| +1 | +2.68 | +1.62 |
| +3 | +1.23 | +0.45 |
A single contrastive direction causally controls the margin (α=+1 drops buggy ~0.98,
scaling with 3B's larger gap). Same mechanism as the paper's 1.5B.

## F5 — monitor (frozen toy v_noop at L32 → 499 SWE prompts)
ROC-AUC **0.867** at the causal cell L32 (projection gap +9.83). Computed server-side via
`modal_app/analyze.py` (no multi-GB download).

**Layer-scan resolves why 0.867 < 1.5B's 0.989** (each layer's own toy contrast direction):
- MAX-AUC layer **L27 (rel 0.750): AUC 0.9771** — pass/fail evidence is as linearly available
  at 3B as at 1.5B (~0.99).
- Causal layer **L32: AUC 0.8668**; **AUC(max) − AUC(causal) = +0.110**.
- So the lower causal-cell AUC is **not** weaker evidence — the max-discriminability layer (L27)
  is simply further from the causal readout (L32) than it was at 1.5B (where L12≈0.998 vs
  L24-causal≈0.989, Δ~0.01). The discriminability-vs-causal-use dissociation **grows with
  scale** (Δ0.110 at 3B). This reinforces the paper's "AUC does not localize the mechanism".

## F6 — action-menu control (position-balanced, 4990 prompts) — **validates F1**
noop argmax rate by menu slot (0=first … 4=last):
| slot | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| overall | 2.0% | 7.0% | 7.3% | 4.4% | 5.5% |
| on **fixed** | 4.0% | 13.8% | 14.6% | 8.8% | 11.0% |
| on **buggy** | 0.0% | 0.2% | 0.0% | 0.0% | 0.0% |

noop is chosen at **every** slot (not just last), **~0% on buggy / 4–15% on fixed everywhere**.
So 3B's abstention is **position-robust and condition-driven by the transcript** — not a
last-slot artifact or an edit-token prior. The F1 break is real.

## Conclusion — the dissociation starts to close at 3B
Evidence is present (F2/F5), causally active (F3 peak L32, F4 steering), **and now partially
acted upon**: 3B abstains ~11% on passing prompts (0% on failing), robustly across menu
positions — vs the paper's 0% at 1.5B. The circuit is preserved (same relative depth) while
the behavior shifts with scale. Static single-turn / first-token caveats still apply.

**Next:** behavioral batch 7B/14B/32B (noop-rate-vs-size curve) — launched detached/spawned.
7B is the width control (same 28 layers as 1.5B).
