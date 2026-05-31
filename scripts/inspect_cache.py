#!/usr/bin/env python3
"""Inspect a cached activation .pt file.

Usage:
    python scripts/inspect_cache.py results/<run_id>/<file>.pt
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(path: Path) -> int:
    import torch  # local-optional dep

    payload = torch.load(path, map_location="cpu", weights_only=False)

    def shape_str(t):
        return "None" if t is None else "x".join(str(s) for s in t.shape)

    print(f"file: {path}")
    print(f"  task / cond / variant : {payload['task_id']} / {payload['condition']} / {payload['variant']}")
    print(f"  model_name            : {payload['model_name']}")
    print(f"  dtype                 : {payload['dtype']}")
    print(f"  seq_len               : {payload['seq_len']}")
    print(f"  last_k                : {payload['last_k']}")
    print(f"  resid_pre             : {shape_str(payload['resid_pre'])}    # (L, B, K, D)")
    print(f"  resid_post            : {shape_str(payload['resid_post'])}   # (L, B, K, D)")
    print(f"  resid_final           : {shape_str(payload['resid_final'])}  # (B, K, D)")
    print(f"  last_k_tokens         : {payload['last_token_text']}")
    print("  action logits:")
    for name, info in payload["action_logits"].items():
        print(
            f"    {name:<5} id={info['first_token_id']:>6}  n_toks={info['n_tokens']}  "
            f"logit={info['logit']:+.3f}  logprob={info['logprob']:+.3f}"
        )
    print("  top-10 next tokens:")
    for row in payload["top_k_next_tokens"]:
        print(f"    {row['token']!r:<20} logit={row['logit']:+.3f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
