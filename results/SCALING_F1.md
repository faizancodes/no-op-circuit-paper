# F1 across the Qwen2.5-Coder ladder — abstention is gated by the action prior, not scale

Behavioral first-token argmax over the 499 SWE-bench-Verified-derived `code_tests` paired
prompts (998 per model; 32B = 996 after a 2400-token paired drop). Same frozen prompts at
every size; only the forward pass changes.

| size | dominant hedge | noop overall | noop on **fixed** (passing) | noop on buggy (failing) |
|---|---|---:|---:|---:|
| 1.5B (paper) | grep ~90% | 0% | 0% | 0% |
| 3B  | **edit** ~91% | 5.5% | **11.0%** | 0% |
| 7B  | view 52%      | 0.0% | 0.0% | 0% |
| 14B | view 46%      | 0.0% | 0.0% | 0% |
| 32B | **edit** 42%  | 2.5% | **5.0%** | 0% |

## Reading
The noop (do-nothing) rate is **non-monotonic** in model size. It is non-zero exactly for the
two sizes whose first-token prior is **`edit`-dominant** (3B, 32B), and zero for the
`grep`/`view`-dominant sizes (1.5B, 7B, 14B). In every non-zero case abstention appears
**only on passing/fixed prompts** (~0% on failing) — the transcript evidence is what tips it.

**Interpretation.** Whether the represented pass/fail evidence *surfaces as an abstention
action* depends on the model's idiosyncratic first-token action prior:
- when the prior concentrates on `edit` (the action that competes with `noop` in the
  `edit − noop` margin), a passing transcript can push the margin far enough that `noop` wins
  on a fraction of prompts;
- when the prior concentrates on `view`/`grep` (investigation actions, outside the edit/noop
  contest), `noop` never wins regardless — the evidence still moves the submargin but cannot
  overcome the prior, exactly the paper's 1.5B dissociation.

So the paper's **0% abstention at 1.5B is not a universal property and not a clean monotonic
function of scale** — it is contingent on 1.5B's `grep` prior. The represented-evidence-vs-action
gap is real at every size, but whether it *closes* depends on the (non-monotonic) action prior,
which the 3B action-menu control already showed is the gating variable (noop ~0% on buggy /
4–15% on fixed at *every* menu position for 3B).

## Position-balanced controls (7B/14B) — resolve the view-dominance
Cyclic orderings placing `noop` in each slot (4,990 prompts/size):
| | noop by slot 0–4 | first-position bias | argmax (balanced) |
|---|---|---|---|
| 7B  | 0.0/0.7/0.0/0.0/0.0% | 34% | view 38% · test 29% · edit 30% · noop 0.1% |
| 14B | 0.0/0.2/0.0/0.0/0.0% | 37% | test 49% · edit 29% · view 20% · noop 0.04% |

- **noop is position-robust at ~0%** for both — non-abstention at 7B/14B is genuine, not a
  menu-position artifact (contrast 3B: 4–15% on fixed at *every* slot).
- The canonical **view-dominance was partly a first-position effect**: under balancing 7B's
  "view 52%" → 38%, and 14B's "view 46%" → **test 49%**. First-position bias is real but modest
  (34–37% vs 20% chance).
- **Refined gating rule:** abstention appears only when the prior is *robustly* edit-dominant —
  3B is (88% edit even balanced), 7B/14B are not (view/test/edit spread). So whether represented
  evidence surfaces as `noop` is gated by the action-prior's *shape*, confirmed not to be a
  position artifact.

## Caveats / next checks
- Causal peak (F3) only measured for 3B (L32, rel-depth 0.889 ✓ predicted). The relative-depth
  law across 7B/14B/32B still needs their patch grids.
- Monitor AUC only for 3B (0.867, below 1.5B's 0.989) — a layer scan / fine-grid peak may lift it.
- Static single-turn, first-token argmax — not a live agent loop.
