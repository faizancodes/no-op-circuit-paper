"""Modal function: characterise the top SAE features that decompose v_noop.

For each feature in `v_noop_features.json::top_features` we report:
  1. Top-activating tokens — stream a small held-out Python slice, run forward
     pass at L24 resid_pre, encode with the SAE, find the top-20 positions
     by activation on that feature. Save (token, context, activation).
  2. Logit-lens — project the feature's decoder vector through Qwen's
     `lm_head.weight` (unembed). Report top-10 promoted and top-10 suppressed
     vocab tokens.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .common import (
    ACTIVATIONS_DIR,
    DEFAULT_GPU,
    DEFAULT_MODEL,
    HF_CACHE_DIR,
    activations_vol,
    app,
    hf_cache_vol,
)

_CHAR_TIMEOUT_S = 60 * 60


@app.function(
    gpu=DEFAULT_GPU,
    timeout=_CHAR_TIMEOUT_S,
    volumes={
        HF_CACHE_DIR: hf_cache_vol,
        ACTIVATIONS_DIR: activations_vol,
    },
)
def characterise_sae_features(
    feature_indices: list[int],
    *,
    sae_path: str = "sae/qwen_l24_resid_pre_d24576_k32.pt",
    model_name: str = DEFAULT_MODEL,
    layer: int = 24,
    n_eval_positions: int = 500_000,
    skip_first_docs: int = 5000,  # offset past the SAE training slice
    max_seq_len: int = 2048,
    dtype: str = "bfloat16",
    top_token_n: int = 20,
    logit_lens_top_n: int = 10,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from no_op_circuit.interp.sae import SAEConfig, TopKSAE

    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]

    t0 = time.time()
    print(f"[characterise] loading {model_name} ({dtype})…", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=HF_CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=HF_CACHE_DIR,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()

    sae_full = Path(ACTIVATIONS_DIR) / sae_path
    blob = torch.load(sae_full, map_location="cpu", weights_only=False)
    cfg = SAEConfig(**blob["config"])
    sae = TopKSAE(cfg).to(device, dtype=torch.float32)
    sae.load_state_dict(blob["state_dict"])
    sae.eval()
    print(f"[characterise] loaded SAE: d_in={cfg.d_in} d_sae={cfg.d_sae} k={cfg.k}", flush=True)

    feature_indices = [int(i) for i in feature_indices]
    print(f"[characterise] feature indices: {feature_indices}", flush=True)

    # ---- Logit-lens via the LM unembed ----
    lm_head_w = model.get_output_embeddings().weight.detach().to("cpu", dtype=torch.float32)  # (V, D)
    W_dec = sae.W_dec.detach().to("cpu", dtype=torch.float32)  # (d_sae, d_in)
    logit_lens_results: dict[int, dict[str, Any]] = {}
    for idx in feature_indices:
        dec_vec = W_dec[idx]                            # (d_in,)
        scores = lm_head_w @ dec_vec                    # (V,)
        prom = scores.topk(logit_lens_top_n)
        supp = (-scores).topk(logit_lens_top_n)
        logit_lens_results[idx] = {
            "top_promote": [
                {"token_id": int(i), "token": tok.decode([int(i)]), "logit": float(scores[i].item())}
                for i in prom.indices.tolist()
            ],
            "top_suppress": [
                {"token_id": int(i), "token": tok.decode([int(i)]), "logit": float(scores[i].item())}
                for i in supp.indices.tolist()
            ],
        }
    print(f"[characterise] logit-lens done", flush=True)

    # ---- Top-activating tokens ----
    print(f"[characterise] streaming held-out corpus (skip {skip_first_docs} docs)…", flush=True)
    try:
        ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
    except Exception:
        ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    ds = ds.skip(skip_first_docs)

    # Heap of (activation, token_id, prev_context_ids, feature_idx)
    import heapq
    per_feature_heaps: dict[int, list] = {idx: [] for idx in feature_indices}
    total_positions = 0
    n_docs = 0

    for doc in ds:
        if total_positions >= n_eval_positions:
            break
        content = doc.get("content") or doc.get("text") or ""
        if not content or not isinstance(content, str):
            continue
        enc = tok(
            content,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 8:
            continue
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        resid = out.hidden_states[layer][0].to(dtype=torch.float32)  # (seq, d)
        a = sae.encode(resid)                                          # (seq, d_sae)
        # For each tracked feature, get this doc's per-position activations
        for idx in feature_indices:
            acts = a[:, idx]
            top_local = acts.topk(min(top_token_n, acts.shape[0]))
            for v, pos in zip(top_local.values.tolist(), top_local.indices.tolist()):
                if v == 0.0:
                    continue
                token_id = int(input_ids[0, pos].item())
                ctx_start = max(0, pos - 5)
                prev_ids = input_ids[0, ctx_start: pos + 1].tolist()
                rec = (float(v), token_id, prev_ids)
                if len(per_feature_heaps[idx]) < top_token_n:
                    heapq.heappush(per_feature_heaps[idx], rec)
                elif rec[0] > per_feature_heaps[idx][0][0]:
                    heapq.heapreplace(per_feature_heaps[idx], rec)
        total_positions += int(input_ids.shape[1])
        n_docs += 1
        if n_docs % 100 == 0:
            print(f"[characterise]   {n_docs} docs, {total_positions:,} positions", flush=True)

    print(f"[characterise] done: {n_docs} docs, {total_positions:,} positions", flush=True)

    # Materialise
    per_feature_results: dict[int, dict[str, Any]] = {}
    for idx in feature_indices:
        items = sorted(per_feature_heaps[idx], key=lambda r: -r[0])
        per_feature_results[idx] = {
            "feature_idx": idx,
            "top_activating": [
                {
                    "activation": float(v),
                    "token": tok.decode([token_id]),
                    "token_id": int(token_id),
                    "context": tok.decode(prev_ids),
                }
                for v, token_id, prev_ids in items
            ],
            "logit_lens_top_promote": logit_lens_results[idx]["top_promote"],
            "logit_lens_top_suppress": logit_lens_results[idx]["top_suppress"],
            "decoder_norm": float(W_dec[idx].norm()),
        }

    out_path = Path(ACTIVATIONS_DIR) / "sae" / "feature_characterisations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"features": per_feature_results,
                                     "n_eval_positions": total_positions,
                                     "n_eval_docs": n_docs}, indent=2))
    activations_vol.commit()
    print(f"[characterise] wrote {out_path}", flush=True)
    print(f"[characterise] elapsed: {time.time()-t0:.1f}s", flush=True)

    return {
        "n_features": len(feature_indices),
        "n_eval_positions": total_positions,
        "n_eval_docs": n_docs,
        "out_path": str(out_path.relative_to(ACTIVATIONS_DIR)),
    }


@app.local_entrypoint()
def characterise_entry(
    features_json: str = "results/sae/v_noop_features_DISTRIBUTED.json",
    sae_path: str = "sae/qwen_l24_resid_pre_TASK_d4096_k16.pt",
    top_n: int = 20,
    model: str = DEFAULT_MODEL,
    layer: int = 24,
    out_suffix: str = "_DISTRIBUTED",
):
    blob = json.loads(Path(features_json).read_text())
    # Prefer the explicit characterisation-target list when present.
    feats = blob.get("top_features_for_characterisation") or blob["top_features"]
    feature_indices = [int(f["feature_idx"]) for f in feats[:top_n]]
    print(f"[entry] characterising {len(feature_indices)} features  "
          f"model={model} layer={layer} sae={sae_path}")
    result = characterise_sae_features.remote(
        feature_indices, sae_path=sae_path, model_name=model, layer=layer,
    )
    print(f"[done] {result}")

    # Pull file local. Always lands at sae/feature_characterisations.json on the
    # volume (function hardcodes this); we save locally with the requested suffix
    # so different (model, sae) runs don't overwrite each other.
    import subprocess
    out_path = result.get("out_path") if isinstance(result, dict) else None
    if out_path:
        local_dir = Path("results/sae")
        local_dir.mkdir(parents=True, exist_ok=True)
        local_target = local_dir / f"feature_characterisations{out_suffix}.json"
        try:
            subprocess.run(
                ["modal", "volume", "get", "noop-activations",
                 out_path, str(local_target), "--force"],
                check=True,
            )
            print(f"[entry] downloaded to {local_target}")
        except Exception as e:
            print(f"[entry] WARN download failed: {e}")
