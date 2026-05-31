# G.17 held-out paraphrase count audit

Per-model held-out counts from `*_scores.json` (CPU, no model forwards).

| model | held-out unique tasks | template A tasks | template B tasks | total prompts | token-cap/missing drops |
|---|---:|---:|---:|---:|---:|
| qwen | 499 | 250 | 249 | 998 | 0 |
| codegemma | 499 | 250 | 249 | 998 | -2 |
| deepseek | 499 | 250 | 249 | 998 | 0 |

## Pooled composition

- **qwen**: 499 paired tasks split across 2 held-out templates (250, 249 tasks); 998 total prompts; pooled AUC pools all templates
- **codegemma**: 499 paired tasks split across 2 held-out templates (250, 249 tasks); 998 total prompts; pooled AUC pools all templates
- **deepseek**: 499 paired tasks split across 2 held-out templates (250, 249 tasks); 998 total prompts; pooled AUC pools all templates
