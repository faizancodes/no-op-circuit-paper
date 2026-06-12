# Action-tokenization audit

Exact scored form = first token(s) after the final `Action: ` (diff prefix vs prefix+name), matching the action-logit readout.

| model | action | n_tokens | single-token? | scored decoded |
|---|---|---:|---|---|
| qwen_0.5b | view | 1 | yes | ` view` |
| qwen_0.5b | grep | 1 | yes | ` grep` |
| qwen_0.5b | test | 1 | yes | ` test` |
| qwen_0.5b | edit | 1 | yes | ` edit` |
| qwen_0.5b | noop | 1 | yes | ` noop` |
| qwen_1.5b | view | 1 | yes | ` view` |
| qwen_1.5b | grep | 1 | yes | ` grep` |
| qwen_1.5b | test | 1 | yes | ` test` |
| qwen_1.5b | edit | 1 | yes | ` edit` |
| qwen_1.5b | noop | 1 | yes | ` noop` |
| qwen_3b | view | 1 | yes | ` view` |
| qwen_3b | grep | 1 | yes | ` grep` |
| qwen_3b | test | 1 | yes | ` test` |
| qwen_3b | edit | 1 | yes | ` edit` |
| qwen_3b | noop | 1 | yes | ` noop` |
| qwen_7b | view | 1 | yes | ` view` |
| qwen_7b | grep | 1 | yes | ` grep` |
| qwen_7b | test | 1 | yes | ` test` |
| qwen_7b | edit | 1 | yes | ` edit` |
| qwen_7b | noop | 1 | yes | ` noop` |
| qwen_14b | view | 1 | yes | ` view` |
| qwen_14b | grep | 1 | yes | ` grep` |
| qwen_14b | test | 1 | yes | ` test` |
| qwen_14b | edit | 1 | yes | ` edit` |
| qwen_14b | noop | 1 | yes | ` noop` |
| qwen_32b | view | 1 | yes | ` view` |
| qwen_32b | grep | 1 | yes | ` grep` |
| qwen_32b | test | 1 | yes | ` test` |
| qwen_32b | edit | 1 | yes | ` edit` |
| qwen_32b | noop | 1 | yes | ` noop` |
