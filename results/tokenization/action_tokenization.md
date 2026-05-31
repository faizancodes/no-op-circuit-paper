# Action-tokenization audit

Exact scored form = first token(s) after the final `Action: ` (diff prefix vs prefix+name), matching the action-logit readout.

| model | action | n_tokens | single-token? | scored decoded |
|---|---|---:|---|---|
| qwen | view | 1 | yes | ` view` |
| qwen | grep | 1 | yes | ` grep` |
| qwen | test | 1 | yes | ` test` |
| qwen | edit | 1 | yes | ` edit` |
| qwen | noop | 1 | yes | ` noop` |
| codegemma | view | 1 | yes | ` view` |
| codegemma | grep | 1 | yes | ` grep` |
| codegemma | test | 1 | yes | ` test` |
| codegemma | edit | 1 | yes | ` edit` |
| codegemma | noop | 1 | yes | ` noop` |
| deepseek | view | 1 | yes | `view` |
| deepseek | grep | 2 | **no** | `grep` |
| deepseek | test | 1 | yes | `test` |
| deepseek | edit | 1 | yes | `edit` |
| deepseek | noop | 2 | **no** | `noop` |
