# Cross-model held-out operating-point calibration (paraphrase setting)

Threshold fit on TRAIN-template prompts (balanced accuracy), applied frozen to 
HELD-OUT-template prompts. Source: `results/heldout_paraphrase_robustness/{model}_scores.json`. CPU only, no model forwards.

| model | train thr | held-out AUC | held-out precision | held-out recall | held-out fixed-cond FPR | held-out acc |
|---|---:|---:|---:|---:|---:|---:|
| qwen | -1.637 | 0.943 | 0.868 | 0.900 | 0.136 | 0.882 |
| codegemma | +5.643 | 0.649 | 0.609 | 0.571 | 0.367 | 0.602 |
| deepseek | -14.082 | 0.619 | 0.611 | 0.611 | 0.389 | 0.611 |

Held-out AUC sanity-check vs G.17 reported: qwen 0.943 (reported 0.943), codegemma 0.649 (reported 0.649), deepseek 0.619 (reported 0.619).

Note: paraphrase-setting held-out calibration; NOT the §5.1 canonical-menu
calibration (raw §5.1 per-task caches are not retained locally).
