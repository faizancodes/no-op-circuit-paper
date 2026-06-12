# Statistical rigor pass (bootstrap CIs, held-out veto, flip-point)

All from existing data; bootstrap B=10,000, seed 0; CIs are 95% percentile.

## F1 — noop (do-nothing) rate on passing prompts

| size | noop on fixed | 95% CI |
|---|---|---|
| 1.5B | 0.0% | (paper) |
| 3B | 11.0% | [8.4, 13.8] |
| 7B | 0.0% | [0.0, 0.0] |
| 14B | 0.0% | [0.0, 0.0] |
| 32B | 5.0% | [3.2, 7.0] |

## F3 — causal patch peak (bidirectional-min), effect CI + sign test

| size | peak L/pos | rel-depth | F→B mean [95% CI] | Wilcoxon p(>0) | peak-layer stability |
|---|---|---|---|---|---|
| 3B | L32/-1 | 0.889 | +1.00 [+0.86, +1.15] | 9.1e-10 | L32 in 92% of boots |
| 7B | L26/-1 | 0.929 | +2.24 [+1.96, +2.54] | 5.5e-10 | L26 in 100% of boots |
| 14B | L44/-1 | 0.917 | +4.09 [+3.53, +4.66] | 5.5e-10 | L44 in 100% of boots |
| 32B | L60/-1 | 0.938 | +11.66 [+11.07, +12.25] | 5.5e-10 | L60 in 100% of boots |

## Steering flip-point (3B, passing prompts)

- noop becomes majority on passing prompts at **α = +6.0** (0%→51%); mean edit−noop margin crosses 0 at **α ≈ 3.99**.
- On buggy prompts noop stays low until higher α (evidence-tracking asymmetry; see KEYSTONE).

## Binary {edit,noop} — noop on passing, with CI

| size | noop on fixed | 95% CI |
|---|---|---|
| 1.5B | 1.4% | [0.7, 2.2] |
| 3B | 42.8% | [39.2, 46.4] |
| 7B | 11.9% | [9.3, 14.7] |
| 14B | 0.0% | [0.0, 0.0] |
| 32B | 0.0% | [0.0, 0.0] |

## Agent loop — over-editing CIs, held-out veto, evidence-gathering cut


### Qwen2.5-Coder-14B-Instruct/issue
- over-editing (passing→edit): **75.0%** [60.0, 87.5]; correct-edit recall 100.0%
- evidence-gathering cut: over-editing among **tested** loops 73% (n=37) vs **untested** 100% (n=3)

### Qwen2.5-Coder-14B-Instruct/ev
- over-editing (passing→edit): **100.0%** [100.0, 100.0]; correct-edit recall 100.0%
- monitor-veto AUC(passing-edit vs buggy-edit) = **0.754** [0.640, 0.851]
- **held-out (LOO) veto**: over-edits BLOCKED **50%** [35,65]; correct-edits PRESERVED **90%** [80,98]
- => over-editing **100% → 50%** (held-out threshold)

### Qwen2.5-Coder-1.5B-Instruct/issue
- over-editing (passing→edit): **35.0%** [20.0, 50.0]; correct-edit recall 35.0%
- evidence-gathering cut: over-editing among **tested** loops nan% (n=0) vs **untested** 35% (n=40)

### Qwen2.5-Coder-1.5B-Instruct/ev
- over-editing (passing→edit): **50.0%** [35.0, 65.0]; correct-edit recall 47.5%
- monitor-veto AUC(passing-edit vs buggy-edit) = **0.676** [0.497, 0.832]
- **held-out (LOO) veto**: over-edits BLOCKED **55%** [35,75]; correct-edits PRESERVED **79%** [58,95]
- => over-editing **50% → 22%** (held-out threshold)

### Qwen2.5-Coder-32B-Instruct/issue
- over-editing (passing→edit): **75.0%** [60.0, 87.5]; correct-edit recall 97.5%
- evidence-gathering cut: over-editing among **tested** loops 78% (n=27) vs **untested** 69% (n=13)

### Qwen2.5-Coder-32B-Instruct/ev
- over-editing (passing→edit): **90.0%** [80.0, 97.5]; correct-edit recall 100.0%
- monitor-veto AUC(passing-edit vs buggy-edit) = **0.800** [0.691, 0.897]
- **held-out (LOO) veto**: over-edits BLOCKED **81%** [67,92]; correct-edits PRESERVED **60%** [45,75]
- => over-editing **90% → 18%** (held-out threshold)

### Qwen2.5-Coder-3B-Instruct/issue
- over-editing (passing→edit): **77.5%** [65.0, 90.0]; correct-edit recall 80.0%
- evidence-gathering cut: over-editing among **tested** loops 0% (n=1) vs **untested** 79% (n=39)

### Qwen2.5-Coder-3B-Instruct/ev
- over-editing (passing→edit): **87.5%** [75.0, 97.5]; correct-edit recall 90.0%
- monitor-veto AUC(passing-edit vs buggy-edit) = **0.892** [0.801, 0.964]
- **held-out (LOO) veto**: over-edits BLOCKED **77%** [63,91]; correct-edits PRESERVED **94%** [86,100]
- => over-editing **88% → 20%** (held-out threshold)

### Qwen2.5-Coder-3B-Instruct/ev_stop
- over-editing (passing→edit): **15.0%** [5.0, 27.5]; correct-edit recall 90.0%

### Qwen2.5-Coder-7B-Instruct/issue
- over-editing (passing→edit): **77.5%** [65.0, 90.0]; correct-edit recall 100.0%
- evidence-gathering cut: over-editing among **tested** loops 47% (n=17) vs **untested** 100% (n=23)

### Qwen2.5-Coder-7B-Instruct/ev
- over-editing (passing→edit): **97.5%** [92.5, 100.0]; correct-edit recall 100.0%
- monitor-veto AUC(passing-edit vs buggy-edit) = **0.787** [0.674, 0.883]
- **held-out (LOO) veto**: over-edits BLOCKED **72%** [56,85]; correct-edits PRESERVED **75%** [60,88]
- => over-editing **98% → 27%** (held-out threshold)

### Qwen2.5-Coder-7B-Instruct/ev_stop
- over-editing (passing→edit): **0.0%** [0.0, 0.0]; correct-edit recall 7.5%

## Ladder — over-editing & veto vs size

| size | over-edit (from-issue) | over-edit (evidence) | veto→ (held-out) | recall (evidence) | explicit-stop: over-edit / recall |
|---|---|---|---|---|---|
| 1.5B | 35% | 50% | 22% | 48% | — |
| 3B | 78% | 88% | 20% | 90% | 15% / 90% |
| 7B | 78% | 98% | 27% | 100% | 0% / 8% |
| 14B | 75% | 100% | 50% | 100% | — |
| 32B | 75% | 90% | 18% | 100% | — |
