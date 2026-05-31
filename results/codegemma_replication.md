# CodeGemma-7b-it cross-model replication

## Setup
- Model: google/codegemma-7b-it (n_layers=28, hidden_size=3072)
- Same 49 paired tasks; same `code_tests` variant; same patching grid
  - Gemma chat template lacks a `system` role; we fold system → first user
    turn via a runtime fallback in `no_op_circuit.agent.prompt`.
- GPU: A10G (T4's 16 GB is too tight for 7B in bf16).
- Cache run: cache-codegemma_7b_it-20260516T031036Z
- Patch run: patch-codegemma_7b_it-20260516T031403Z
- Steer run: skipped (peak per-task universality fell below the gate; see
  Phase 4 verdict)
- Grid: `--max-suffix 2 --layer-step 2 --bidirectional` (matches the Qwen
  bidirectional run exactly; 14 layers {0, 2, …, 26} × 2 positions × 2
  directions per pair).

## Behavioral Δ-margin (cache only)
| variant | n | mean Δ | median Δ | positive |
|---|---|---|---|---|
| issue_only | 49 | +0.000 | +0.000 | 0/49 |
| code | 49 | -0.082 | +0.000 | 9/49 |
| code_tests | 49 | **+1.347** | **+2.000** | **29/49** |

Qwen reference: code_tests mean Δ = +0.659, 47/49 positive.

CodeGemma's mean Δ is roughly **2× Qwen's**, and its median is +2.0 — but
positive coverage drops to 59% (vs Qwen's 96%). The distribution is bimodal:
many tasks show a strong shift, the rest show ~0. The phenomenon is present
and arguably stronger when it fires, but less universal.

## Bidirectional patching peak
| metric | CodeGemma | Qwen |
|---|---|---|
| Peak (layer, pos) | **L26, pos −1** | L24, pos −1 |
| Relative depth (peak / n_layers) | **26/28 = 0.929** | 24/28 = 0.857 |
| F→B mean shift at peak | **+1.143** | +0.688 |
| F→B median shift at peak | **+2.000** | +0.688 |
| F→B % positive at peak | **53%** | 100% |
| B→F mean shift at peak | **+1.184** | +0.640 |
| B→F median shift at peak | **+2.000** | +0.640 |
| B→F % positive at peak | **61%** | 100% |
| min(F→B, B→F) at peak | **+1.143** | +0.640 |
| L0 sanity (must be 0) | +0.000 / +0.000 | +0.000 / +0.000 |

The single-cell mean shift is ≈ 1.8× Qwen's, but the per-task universality
is ~57% (vs Qwen 100%). On the tasks where patching fires, it fires
*harder* (median = +2.0 logits, not +0.69). When it doesn't fire, it
fires not at all — consistent with a binary "saturate-or-shift" response.

## Layer × position heatmap (F→B mean shift)
```
layer p= -2  p= -1
L0   +0.000  +0.000
L2   +0.122  +0.082
L4   +0.122  +0.041
L6   +0.000  +0.122
L8   +0.327  +0.327
L10  +0.531  +0.408
L12  +0.327  +0.367
L14  +0.408  +0.408
L16  +0.367  +0.449
L18  +0.122  +0.980
L20  +0.449  +0.776
L22  +0.163  +0.898
L24  +0.204  +0.612
L26  +0.041  +1.143
```

The signal builds non-monotonically: early-mid layers (L8–L14) carry a
modest shift at both positions, then concentrates on pos −1 from L18
onward, peaking at L26. Position −2 has its own modest peak at L10 (+0.53)
that's absent in Qwen. This suggests CodeGemma may carry the no-op
information across slightly more positions than Qwen, which kept almost
everything at pos −1.

## Layer × position heatmap (B→F mean shift) — bidirectional symmetry
```
layer p= -2  p= -1
L0   +0.000  +0.000
L2   +0.286  +0.204
L4   +0.367  +0.163
L6   +0.286  +0.245
L8   +0.408  +0.245
L10  +0.367  +0.204
L12  +0.286  +0.327
L14  +0.408  +0.245
L16  +0.122  +0.653
L18  +0.531  +1.429   ← B→F peak
L20  +0.408  +1.184
L22  +0.163  +0.939
L24  +0.204  +0.653
L26  +0.204  +1.184
```

In contrast to Qwen (where F→B and B→F peaked at the exact same cell), the
CodeGemma B→F peak is at **L18**, not L26. The F→B peak is L26. They
disagree by 8 layers. The min(F→B, B→F) ranking still puts L26/pos −1 at
the top because both directions are strong there, but the *primary*
direction is asymmetric — F→B prefers very late (L26), B→F prefers
deeper-mid (L18). This is a non-trivial departure from Qwen's clean
symmetry.

## Steering dose-response
SKIPPED. Gate required ≥ 80% positive in BOTH directions at the peak; we
observed 53% (F→B) and 61% (B→F). A steering sweep on the full corpus
would conflate the saturating-half (no effect) with the responsive-half
(strong effect) and produce a misleadingly flat curve. A targeted sweep
on the responsive subset is a follow-up.

## Verdict

**PARTIAL**: The behavioral phenomenon replicates more strongly in absolute
terms (mean Δ = +1.347 vs Qwen +0.659) and the patching peak is at the
expected very-late position (L26 / pos −1, relative depth 0.929 vs Qwen
0.857), but the per-task universality drops sharply (53–61% positive vs
Qwen's 100%) and the F→B / B→F asymmetry between L26 and L18 breaks the
clean bidirectional symmetry we found in Qwen. The no-op mechanism appears
to exist in CodeGemma but is **sparser** — concentrated on a subset of
tasks where it carries the *entire* decision (median shift +2.0 logits
when it fires) and absent on the rest.

Implication for the paper: the "single localizable bidirectional direction"
finding generalizes across architectures in its existence and its
late-layer concentration, but its **universality** is a Qwen-favourable
property. The honest framing is *the no-op direction is real and
late-residual in both models; CodeGemma's instruction-tuning produces a
binary on/off response per task while Qwen's produces a continuous
gradient*. That's a finding about how IT regimes shape mechanistic
abstention, not a failure of replication.
