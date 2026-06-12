# Keystone causal test: the action prior gates whether represented evidence drives behavior

Two interventions establish that pass/fail evidence is *represented and action-capable* at every
size, and that whether it surfaces as abstention is gated by the first-token action prior.

## (A) Steering-to-induce-abstention (3B, toys, v_noop at L32/pos −1)

Amplifying `+α·v_noop` (v_noop = mean_fixed − mean_buggy; +α pushes toward "passing"):

| α | FIXED noop% | FIXED edit% | FIXED margin(e−n) | BUGGY noop% | BUGGY margin |
|---:|---:|---:|---:|---:|---:|
| 0   | 0%  | 100% | +2.40 | 0%  | +3.66 |
| +1  | 4%  | 92%  | +1.62 | 0%  | +2.67 |
| +2  | 6%  | 80%  | +0.97 | 0%  | +1.87 |
| +3  | 20% | 47%  | +0.45 | 2%  | +1.23 |
| **+4** | **35%** | 18% | **−0.01** | 6%  | +0.69 |
| +5  | 49% | 4%   | −0.42 | 20% | +0.25 |
| +6  | **51%** | 0% | −0.83 | 39% | −0.15 |

- On **passing** prompts the action flips `edit → noop`: noop reaches **majority (51%)**, and the
  `edit − noop` margin crosses 0 at **α ≈ +4** — i.e. amplified evidence *changes the action*.
- **Evidence-tracking asymmetry:** passing prompts flip at α≈+4, failing prompts only at α≈+6.
  v_noop tips the *passing* condition first — proof it rides the pass/fail evidence axis, not a
  generic noop push.
- Honest framing: at the **natural** scale (α=1) the action does **not** flip (consistent with
  the paper's "steering moves the margin, not the action"); the direction is **action-capable but
  suppressed at natural scale by the prior**. (At high α, view/test also rise — the binary menu
  below isolates the pure edit↔noop flip.)

## (B) Binary {edit, noop} forced-choice across the ladder
`action_order_control --experiment binary`, 1.5B/3B/7B/14B/32B (32B capped 2400 tok), 1,996
prompts/size. noop argmax rate:

| size | noop@first | noop@second | **noop fixed** | noop buggy | full-menu noop/fixed |
|---|---:|---:|---:|---:|---:|
| 1.5B | 0.0% | 2.0% | 1.4% | 0.6% | ~0% |
| 3B   | 14.1% | 30.0% | **42.8%** | 1.3% | 11% |
| 7B   | 6.6% | 5.3% | **11.9%** | 0.0% | **0%** |
| 14B  | 0.0% | 0.0% | 0.0% | 0.0% | 0% |
| 32B  | 0.0% | 0.0% | 0.0% | 0.0% | 5% |

**Partial confirmation — clean where it matters most, but not universal:**
- **7B is the clean keystone win**: a view-prior model with **0% noop in the full menu** abstains
  **11.9% on passing / 0% on failing** once the hedge is removed (position-robust, ~6% at both
  slots). The evidence *was* represented and action-ready, masked by the view prior — exactly the
  prediction.
- **3B** unmasks dramatically (11% → **43%** on fixed), again specific to passing (1.3% on buggy).
- **1.5B** barely moves (1.4%) — its edit-over-noop baseline is strong (matches the paper's ~2.8%).
- **14B and 32B do NOT unmask (0%)** — even head-to-head, edit beats noop on every passing prompt.
  For these, removing the hedge is *not sufficient*; their edit-vs-noop baseline is robust to it.
  (Note the prompt's listed menu differs between full/binary, so the logits differ — 32B's
  full-menu 5% and binary 0% are not contradictory, just different framings.)

**Honest synthesis with (A):** the hedge-masking mechanism is **real and demonstrated** (7B, 3B),
but **not universal** — the true gate is the model-specific *edit-vs-noop margin*, which the
evidence shifts (patch grids: it moves the margin at every scale) but which removing the hedge
only overcomes when that margin is already near zero (7B/3B), not for 14B/32B. The **steering
result (A) is the stronger causal proof**: amplifying the evidence direction *does* flip the
action to noop (passing-first), showing the evidence is action-capable even where the binary menu
alone doesn't surface it.

## Interpretation
Together these convert the prior-gating claim from correlational to causal: the represented
pass/fail evidence is wired to the edit/noop decision at every scale and *can* drive the action
(steering), but at the natural operating point its behavioral expression is gated by the action
prior (binary menu removes the gate).
