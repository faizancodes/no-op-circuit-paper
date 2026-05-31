---
pretty_name: no-op-circuit-caches
license: other
license_name: see-data-use-notes
tags:
  - mechanistic-interpretability
  - activations
  - residual-stream
  - swe-bench
task_categories:
  - other
---

# no-op-circuit-caches

Activation-cache archive for "A Residual Direction for Pass/Fail Transcript Evidence in Static Coding-Agent Prompts."

## Contents

This dataset contains residual-stream activation caches and run manifests used for the monitor, control, and sweep analyses in the paper. It is primarily per-task/per-variant/per-condition `.pt` tensors plus metadata needed to reproduce the aggregate analysis scripts.

Caches are named by experiment, e.g.:

- `cache-20260515T221105Z` — Qwen toy substrate (§4)
- `cache-{codegemma_7b_it,deepseek-toy}-*` — toy substrates for the other two models (§4.3)
- `cache-real-{qwen,codegemma,deepseek}-n500-*` — §5.1 monitor evaluation
- `cache-real-qwen-swap-n500-*` — §5.2 contradictory-transcript control
- `cache-real-{codegemma,deepseek}-paraphrase-*` — App. G.12 paraphrase controls

## Not included

This archive is not a standalone source-code release. The paper's small metric JSONs, frozen direction artifacts, plotting scripts, and analysis scripts are released in the code repository.

## Data-use notes

The caches are derived from SWE-bench-Verified-derived static paired prompts and toy Python tasks. The per-prompt records may include prompt-derived metadata such as last-token text or token ids, which can contain short fragments of oracle code windows. Full upstream source files and full oracle windows are not intentionally redistributed.

## How to download

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "faizancodes/no-op-circuit-caches",
    repo_type="dataset",
    local_dir="results",
)
```

For a single cache:

```bash
hf download faizancodes/no-op-circuit-caches \
    --repo-type dataset \
    --local-dir results \
    --include "cache-real-qwen-n500-20260516T235301Z/**"
```

## License

Use the same license as the source repository unless a different license is explicitly required by the dataset host. The upstream repositories represented in SWE-bench retain their original licenses.
