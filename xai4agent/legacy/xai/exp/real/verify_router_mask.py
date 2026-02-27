#!/usr/bin/env python3
"""
Verify router mask effectiveness on top-k routing (last token).

For each prompt:
  - run forward without mask -> count masked routers in top-k
  - run forward with mask -> count masked routers in top-k (should be 0)
Also record max masked logit after mask (should be ~ -1e9).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/root/autodl-tmp/models/qwen3-coder-30b"
DEFAULT_DATA = "/root/autodl-tmp/xai/exp/real/real_tool_prompts.json"
DEFAULT_SERVER = "/root/autodl-tmp/FastAPI/qwen3coder/server_intervene.py"
DEFAULT_OUTPUT = "/root/autodl-tmp/xai/exp/real/verify_router_mask.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify router mask via top-k routing.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--server_path", default=DEFAULT_SERVER)
    parser.add_argument("--mask", default=None, help="Mask points string, e.g. \"{(30,24),(34,68)}\"")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--mode", choices=["hook", "mask_logits"], default="hook")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Failed to load module: {path}")
    spec.loader.exec_module(module)
    return module


def normalize_layers(router_logits: Any) -> List[torch.Tensor]:
    if isinstance(router_logits, torch.Tensor):
        if router_logits.dim() == 4:
            return [router_logits[i] for i in range(router_logits.shape[0])]
        return [router_logits]
    return list(router_logits)


def last_token_topk_indices(layer_logits: torch.Tensor, k: int) -> torch.Tensor:
    if layer_logits.dim() == 2:  # [T, E]
        vec = layer_logits[-1]
    elif layer_logits.dim() == 3:  # [B, T, E]
        vec = layer_logits[0, -1]
    elif layer_logits.dim() == 4:  # [B, H, T, E] (unlikely)
        vec = layer_logits[0, 0, -1]
    else:
        raise RuntimeError(f"Unexpected router_logits dim: {layer_logits.dim()}")
    _, idx = torch.topk(vec, k=k, dim=-1)
    return idx


def get_masked_logits(layer_logits: torch.Tensor, routers: List[int]) -> torch.Tensor:
    if layer_logits.dim() == 2:
        vec = layer_logits[-1]
    elif layer_logits.dim() == 3:
        vec = layer_logits[0, -1]
    elif layer_logits.dim() == 4:
        vec = layer_logits[0, 0, -1]
    else:
        raise RuntimeError(f"Unexpected router_logits dim: {layer_logits.dim()}")
    return vec[routers]


def main() -> int:
    args = parse_args()

    data = json.loads(Path(args.data_path).read_text())
    if not isinstance(data, list):
        raise SystemExit("Expected list in data_path.")

    if args.max_samples is not None:
        data = data[: args.max_samples]
    if not data:
        raise SystemExit("No prompts to test.")

    srv = load_module(args.server_path, "srv")
    mask_points = srv._normalize_mask_value(args.mask)
    mask_by_layer = srv._build_layer_router_mask(mask_points)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    # Register hooks
    hooks = []
    for i, layer in enumerate(model.model.layers):
        gate = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
        if gate is not None:
            hooks.append(gate.register_forward_hook(srv._make_gate_hook(i)))

    top_k = int(getattr(model.config, "num_experts_per_tok", 8))

    counts_before: Dict[Tuple[int, int], int] = {pt: 0 for pt in mask_points}
    counts_after: Dict[Tuple[int, int], int] = {pt: 0 for pt in mask_points}
    total_prompts = 0
    max_masked_logit_after = float("-inf")

    for row in data:
        prompt = row.get("prompt")
        if not prompt:
            continue
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(args.device)

        # baseline forward
        token = srv.router_mask_var.set({})
        try:
            with torch.no_grad():
                out_base = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_router_logits=True,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            srv.router_mask_var.reset(token)

        layers_base = normalize_layers(out_base.router_logits)

        if args.mode == "hook":
            token = srv.router_mask_var.set(mask_by_layer)
            try:
                with torch.no_grad():
                    out_mask = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_router_logits=True,
                        use_cache=False,
                        return_dict=True,
                    )
            finally:
                srv.router_mask_var.reset(token)
            layers_mask = normalize_layers(out_mask.router_logits)
        else:
            # Apply mask directly to logits (equivalent to masking before top-k)
            layers_mask = []
            for idx, layer_logits in enumerate(layers_base):
                t = layer_logits.clone()
                routers = mask_by_layer.get(idx)
                if routers:
                    t[..., list(routers)] = -1e9
                layers_mask.append(t)

        for layer_idx, routers in mask_by_layer.items():
            if layer_idx >= len(layers_base):
                continue
            topk_before = last_token_topk_indices(layers_base[layer_idx], top_k).tolist()
            topk_after = last_token_topk_indices(layers_mask[layer_idx], top_k).tolist()

            for router_id in routers:
                if router_id in topk_before:
                    counts_before[(layer_idx, router_id)] += 1
                if router_id in topk_after:
                    counts_after[(layer_idx, router_id)] += 1

            masked_vals = get_masked_logits(layers_mask[layer_idx], list(routers))
            max_masked_logit_after = max(max_masked_logit_after, float(masked_vals.max().item()))

        total_prompts += 1
        if total_prompts % 50 == 0:
            print(f"processed {total_prompts}/{len(data)}")

    # remove hooks
    for h in hooks:
        h.remove()

    result = {
        "total_prompts": total_prompts,
        "mask_points": sorted(list(mask_points)),
        "counts_before": {str(k): v for k, v in counts_before.items()},
        "counts_after": {str(k): v for k, v in counts_after.items()},
        "max_masked_logit_after": max_masked_logit_after,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
