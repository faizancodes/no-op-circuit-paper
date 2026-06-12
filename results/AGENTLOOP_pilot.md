# Agent-loop pilot: over-editing is real, and the monitor works as an edit-veto

Minimal multi-turn loop (`modal_app/agent_loop.py`): faithful decisions, simulated execution.
Each turn the model picks from {view, grep, test, edit, noop}; `view` reveals code, `test`
reveals the pass/fail transcript (from the paired data), `edit`/`noop` terminate. 40 SWE-derived
tasks × 2 conditions (passing = fixed code; buggy = needs fix), on 3B (edit-prior) vs 7B
(view-prior). Two start states: **from-issue** (issue + file list only → agent must gather
evidence) and **evidence-present** (code + transcript in context from turn 1).

## Over-editing (fraction of already-passing tasks the agent still edits)
| model | from-issue | evidence-present | tested (from-issue) |
|---|---:|---:|---:|
| 3B | **77.5%** | **87.5%** | 2% |
| 7B | **77.5%** | **97.5%** | 42% |

- **Over-editing is severe and real** — both models edit already-correct code 77–98% of the time.
- **Having the pass/fail evidence in context does NOT reduce it** — over-editing is *higher* with
  evidence present (3B 77→88%, 7B 77→98%). The paper's "evidence represented but doesn't change
  the action" manifests directly as real over-editing in a loop.
- **Evidence-gathering is a second bottleneck**: 3B almost never tests (2%) — it edits blind; 7B
  tests more (42%) and has perfect recall (100% edits on buggy) but still over-edits 77%.

## Monitor-as-edit-veto (read resid·v_noop at the decision; veto edit if it reads "passing")
| model · mode | veto AUC | over-edits blocked | correct-edits preserved | over-editing → | recall → |
|---|---:|---:|---:|---:|---:|
| 3B · from-issue | 0.512 | 10% | 94% | 78%→70% | 80%→75% |
| **3B · evidence-present** | **0.892** | **77%** | **97%** | **88%→20%** | **90%→88%** |
| 7B · from-issue | 0.722 | 45% | 92% | 78%→42% | 100%→92% |
| **7B · evidence-present** | **0.787** | **74%** | **78%** | **98%→25%** | **100%→78%** |

- **The veto works when the transcript is in context.** Evidence-present: it cuts over-editing
  from ~88–98% down to **20–25%** while preserving **78–88%** of correct edits. For 3B it's
  excellent (20% over-editing at 88% recall); for 7B the recall cost is higher (78%).
- **The veto is useless without gathered evidence** (3B from-issue AUC 0.512 ≈ chance — only 2%
  tested, so the monitor has no signal). This is the honest bound: the veto needs the agent to
  have run the tests.

## What this establishes for the paper
1. The static finding is **not an artifact** — it produces real, severe over-editing in a loop,
   and the represented evidence's failure to change the action is exactly what drives it.
2. The mechanism is **deployable as an edit-veto**: the internal pass/fail direction, read at the
   action position, cuts over-editing 3–4× at high recall **when the evidence is in context**.
3. Two bottlenecks to honest deployment: (a) the agent must gather the test evidence (3B barely
   does), (b) the veto's recall cost is model-dependent.

## Caveats
- Pilot scale (N=40), simulated execution (transcripts synthesized, no real sandbox), greedy
  menu-constrained action selection (not free-form generation). Veto thresholds are **held-out**
  (leave-one-out; see the full-ladder section) — the relative over-editing reduction is the robust
  signal, the exact percentages will move with scale.

## Full-ladder extension (1.5B/14B/32B added) + robustness

**Over-editing & held-out veto vs size** (evidence-present; veto threshold leave-one-out):
| size | over-edit (issue) | over-edit (evidence) | + veto → | recall (evidence) | veto AUC |
|---|---:|---:|---:|---:|---:|
| 1.5B | 35% | 50% | 22% | 48%* | 0.676 |
| 3B | 78% | 88% | **20%** | 90% | 0.892 |
| 7B | 78% | 98% | **27%** | 100% | 0.787 |
| 14B | 75% | 100% | 50% | 100% | 0.754 |
| 32B | 75% | 90% | **18%** | 100% | 0.800 |

\* 1.5B is the lone low-over-editing point only because it is *indecisive* (hits max turns
investigating, not abstaining) and its recall is poor (48%). For 3B–32B over-editing is severe
(75–100%), and giving the passing transcript in context does **not** reduce it (14B edits 100%
of passing tasks). The held-out monitor-veto cuts over-editing **2–5×** at every scale, recall
preserved 60–94% (model-dependent; 3B cleanest at 94%, 32B blocks hardest at 60%).

**Robustness — explicit "stop if tests pass" system prompt (3B/7B, evidence-present):**
- 3B: over-editing 88%→**15%**, recall **90%** preserved — follows the instruction well.
- 7B: over-editing 98%→**0%**, but recall collapses to **8%** — it noops *everything*, losing the
  ability to edit when needed.
So a blunt prompt instruction is **brittle**: it either under-uses abstention (default) or
over-applies it and destroys recall (7B). The evidence-conditioned veto cuts over-editing while
*preserving* recall — a targeted internal signal beats a blunt instruction. (And in the realistic
from-issue mode the instruction is moot for models that do not gather the evidence: 3B tests 2%.)
