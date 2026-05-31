# Qwen five-action reconciliation (Experiment A)

**Status: RECONCILIATION INCOMPLETE — §5.6 is a current-renderer sanity check.**

## Availability
- Local Qwen patch artifact: `results/patch-qwen25_coder_15b_instruc-20260516T051331Z/`
  is the **`code` negative-control variant only** (no toy `code_tests` patch grid).
- HF dataset `faizancodes/no-op-circuit-caches`: holds only SWE-derived residual
  caches (`cache-real-qwen-n500-*`, `cache-real-qwen-swap-n500-*`) and `sae/`;
  **no toy `code_tests` patch artifact**.
- Therefore the §4.1 toy patch artifact (clean+patched margins behind +0.648/+0.659)
  could not be loaded; a prompt-hash / per-task margin reconciliation was not possible,
  and a GPU rerun would reproduce the current renderer rather than the original.

## Numbers (no merge performed)
| quantity | §4.1/§4.3 (archived renderer) | §5.6 current renderer (this run) |
|---|---|---|
| clean B−F gap (all 49) | +0.659 | +0.532 |
| F→B shift at L24/pos−1 | +0.648 (all-49 one-way) | +0.625 |
| B→F shift at L24/pos−1 | +0.64 (43-task bidir) | +0.596 |

The F→B shift is close (+0.625 vs +0.648);
the clean gap differs (+0.532 vs +0.659),
consistent with fixed-condition renderer drift after May-2026 (same direction as the CodeGemma audit).

## Decision
RECONCILIATION INCOMPLETE — current-renderer sanity check only. The §4.1 toy code_tests PATCH artifact is unavailable (see availability), so per-task prompt-hash / margin reconciliation could not be performed. The current-renderer F->B shift (+0.625) is close to the §4.1 all-49 estimate (+0.648), but the current-renderer clean B-F gap (+0.532) is lower than §4.1's +0.659. Most likely cause: fixed-condition prompt-rendering drift after the May-2026 paper run (same direction observed in the CodeGemma audit). Per the decision rule, §5.6 is treated as a renderer-specific sanity check and its clean-gap column is removed; the §4.1/§4.3 causal-localization numbers are NOT revised.
