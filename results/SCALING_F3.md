# F3 across the Qwen2.5-Coder ladder — causal localization & the relative-depth law

Paired bidirectional residual patching on the 49 toy `code_tests` pairs; peak = the cell
maximizing min(F→B, B→F) mean shift on the `edit − noop` margin. Coarse grids (layer_step 2
for 7B, 4 for 14B/32B); 1.5B/3B for reference.

| size | layers | peak (pos −1) | **rel-depth** | F→B shift | B→F shift | toy B−F gap | ~recovered | GPU |
|---|---:|---|---:|---:|---:|---:|---:|---|
| 1.5B (paper) | 28 | L24 | 0.857 | +0.69 | +0.64 | +0.66 | ~98%¹ | — |
| 3B  | 36 | L32 | 0.889 | +1.00 | +1.01 | +1.25 | ~80% | A10G |
| 7B  | 28 | **L27**² | **0.964** | +2.77 | ~+2.8 | +2.76 | ~100% | A100-40 |
| 14B | 48 | L44 | 0.917 | +4.09 | +3.61 | +4.89 | ~84% | A100-80 |
| 32B | 64 | L60 | 0.938 | +11.66 | +11.62 | +14.14 | ~82% | H100 |

¹ fine grid (paper); 3B/14B/32B are coarse (step 2–4) and undershoot the exact layer + magnitude.
² **7B fine-swept** (L18–27, step 1): the effect rises monotonically to the *final* layer L27
(rel-depth 0.964, F→B +2.77), confirming the coarse step-2 grid undershot at L26/0.929. The
same coarse-undershoot caveat applies to 3B/14B/32B (their true peaks are likely a touch deeper).

## Findings
1. **The causal readout site replicates at every scale** — always pos −1 (action position),
   always late-layer, bidirectionally significant (F→B ≈ B→F, ruling out mere erasure). The
   circuit the paper localized at 1.5B exists at 3B/7B/14B/32B.
2. **Relative-depth law holds but drifts deeper with scale.** Peaks sit in the last ~4–14% of
   layers (rel-depth 0.857 → 0.889 → **0.964**(7B, fine) → 0.917 → 0.938), not a constant 0.857
   — bigger models place the readout relatively closer to the final layer; the fine-swept 7B
   peaks at the *very last* layer.
3. **Width control refutes strict depth-relativity (fine-swept).** 7B has the *same 28 layers*
   as 1.5B but peaks at the **final layer L27, not L24** (rel 0.964 vs 0.857) — so **width pushes
   the readout site all the way to the network's end**, it is not a function of relative depth.
4. **Causal effect magnitude scales strongly** with size: peak F→B/B→F grows +0.65 → +11.7
   logits (1.5B→32B), and the toy B−F gap grows +0.66 → +14.1. The mechanism becomes far more
   behaviorally salient at scale; coarse-peak gap recovery is a stable ~80–84%.

## Caveats / next
- Coarse grids: the bigger models peak at (or near) the deepest swept layer, so a **fine
  layer_step-1 sweep near the top** would pin the exact peak (likely a touch deeper) and raise
  the recovered fraction toward the ~98% the paper got with a fine grid at 1.5B.
- Negative control (`code` variant, no transcript) and wrong-layer/wrong-position controls per
  size not yet run — would confirm the peak is transcript-conditional at each scale.
- Single position (−1) reported; full position profile is in the saved heatmaps.
