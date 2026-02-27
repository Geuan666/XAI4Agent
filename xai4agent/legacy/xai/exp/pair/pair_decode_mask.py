#!/usr/bin/env python3
"""Generate autoregressive completions for paired prompts (without answers).

This script:
1. Uses token_build output to align assisted to agentic positions
2. Removes the answer tokens using canonical fence positions
3. Autoregressively decodes with proper attention_mask
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = "/root/autodl-tmp/xai/exp/pair/pair_prompts.json"
DEFAULT_TOKENS = "/root/autodl-tmp/xai/exp/pair/pair_tokens.json"
DEFAULT_MODEL = "/root/autodl-tmp/models/qwen3-coder-30b"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/xai/output/humaneval/decode_mask"
DEFAULT_LOG = "/root/autodl-tmp/xai/exp/pair/pair_decode_mask.tsv"

MARKER = "#Function definition and docstring:\n"
FALLBACK_MARKER = "Function definition and docstring:\n"
ASSISTANT_TAG = "<|im_start|>assistant"
IM_END = "<|im_end|>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
USER_START = "<|im_start|>user"
SYSTEM_START = "<|im_start|>system"

# Tool-mode router mask points from Image #1 (last token tool-mode top-8)
DEFAULT_ROUTER_MASK_POINTS = {
    (44, 102),
    (38, 81),
    (27, 122),
    (42, 2),
    (45, 44),
    (24, 108),
    (33, 84),
    (40, 64),
    (39, 10),
    (36, 1),
    (32, 26),
    (22, 56),
    (46, 118),
    (30, 8),
    (39, 122),
    (34, 48),
    (41, 23),
    (26, 5),
    (16, 64),
    (35, 11),
    (19, 112),
    (21, 56),
    (32, 57),
    (20, 26),
    (31, 112),
    (41, 18),
    (29, 108),
    (6, 40),
    (24, 3),
    (5, 92),
    (40, 17),
    (16, 115),
    (21, 110),
    (2, 26),
    (30, 24),
    (8, 71),
    (23, 49),
    (47, 14),
    (46, 38),
    (9, 27),
    (35, 113),
    (33, 125),
    (39, 12),
    (6, 60),
    (19, 49),
    (45, 14),
    (26, 81),
    (36, 56),
    (13, 46),
    (27, 10),
    (44, 31),
    (47, 77),
    (17, 91),
    (47, 118),
    (37, 122),
    (14, 81),
    (30, 83),
    (31, 71),
    (21, 0),
    (5, 14),
    (36, 3),
    (14, 18),
    (34, 112),
    (12, 3),
    (34, 56),
    (15, 31),
    (11, 80),
    (43, 31),
    (20, 71),
    (40, 58),
    (35, 7),
    (40, 39),
    (33, 16),
    (43, 83),
    (28, 17),
    (12, 108),
    (9, 75),
    (21, 27),
    (13, 114),
    (18, 40),
    (46, 85),
    (25, 116),
    (18, 60),
    (39, 8),
    (29, 21),
    (28, 64),
    (22, 2),
    (25, 28),
    (17, 23),
    (38, 110),
}

# Masking mode:
# - "replace_topk": mask logits then reselect top-k from remaining experts
# - "keep_topk": keep only original top-k, then mask within them and renormalize
DEFAULT_MASK_MODE = "keep_topk"


def _build_layer_router_mask(points: set[tuple[int, int]]) -> dict[int, set[int]]:
    mask: dict[int, set[int]] = {}
    for layer_idx, router_idx in points:
        if layer_idx < 0 or router_idx < 0:
            continue
        mask.setdefault(layer_idx, set()).add(router_idx)
    return mask


def _validate_router_mask(mask: dict[int, set[int]], num_experts: int, top_k: int) -> None:
    for layer_idx, routers in mask.items():
        if len(routers) >= num_experts:
            raise ValueError(f"Mask removes all experts at layer {layer_idx}")
        if num_experts - len(routers) < top_k:
            raise ValueError(f"Mask leaves fewer than top_k experts at layer {layer_idx}")


def _mask_keep_topk(out: torch.Tensor, router_ids: set[int], top_k: int) -> torch.Tensor:
    # Keep only the original top-k logits, then mask within them.
    orig = out
    k = min(top_k, orig.shape[-1])
    topk_vals, topk_idx = orig.topk(k=k, dim=-1)

    neg_inf = torch.tensor(float("-inf"), device=orig.device, dtype=orig.dtype)
    kept = torch.full_like(orig, neg_inf)
    kept.scatter_(-1, topk_idx, topk_vals)

    if router_ids:
        kept[..., list(router_ids)] = neg_inf

    # If everything got masked (all -inf), fall back to original top-k without masking.
    all_masked = torch.isinf(kept).all(dim=-1, keepdim=True)
    if all_masked.any():
        fallback = torch.full_like(orig, neg_inf)
        fallback.scatter_(-1, topk_idx, topk_vals)
        kept = torch.where(all_masked, fallback, kept)

    return kept


def _make_gate_hook(layer_idx: int, mask: dict[int, set[int]], top_k: int, mask_mode: str):
    def hook(module, inp, out):
        router_ids = mask.get(layer_idx)
        if not router_ids:
            return out
        if not torch.is_tensor(out):
            return out
        if mask_mode == "keep_topk":
            return _mask_keep_topk(out, router_ids, top_k)
        out = out.clone()
        out[..., list(router_ids)] = -1e9
        return out

    return hook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoregressive decode for paired prompts.")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--mask-mode", choices=["replace_topk", "keep_topk"], default=DEFAULT_MASK_MODE)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        import orjson
        return orjson.loads(path.read_bytes())
    except Exception:
        return json.loads(path.read_text())


def resolve_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def remove_answer(prompt: str) -> str:
    """Remove the trailing ```python...``` answer from the prompt."""
    idx = prompt.rfind("```python")
    if idx != -1:
        return prompt[:idx].rstrip()
    return prompt


def list_keys(data: dict[str, Any]) -> list[str]:
    keys = list(data.keys())
    try:
        keys.sort(key=lambda x: int(x.split("_")[1]))
    except Exception:
        keys.sort()
    return keys


def main() -> int:
    args = parse_args()
    prompts_path = Path(args.prompts)
    tokens_path = Path(args.tokens)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_json(prompts_path)
    token_data = load_json(tokens_path)
    keys = list_keys(data)
    start = max(args.start, 0)
    end = len(keys) if args.limit is None else min(len(keys), start + args.limit)

    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(args.device)
    model.eval()

    # Register router-mask hooks (MoE gate)
    router_mask = _build_layer_router_mask(DEFAULT_ROUTER_MASK_POINTS)
    num_experts = getattr(model.config, "num_experts", None)
    top_k = getattr(model.config, "num_experts_per_tok", 8)
    if num_experts is not None:
        _validate_router_mask(router_mask, int(num_experts), int(top_k))
    hooks = []
    try:
        for i, layer in enumerate(model.model.layers):
            gate = getattr(layer.mlp, "gate", None) or getattr(layer.mlp, "router", None)
            if gate is not None:
                hooks.append(gate.register_forward_hook(_make_gate_hook(i, router_mask, int(top_k), args.mask_mode)))
    except Exception as exc:
        raise RuntimeError(f"Failed to register router mask hooks: {exc}")

    log_path.write_text("timestamp\tkey\tvariant\tstatus\tinput_len\toutput_len\tduration_s\tmessage\n")

    for idx in range(start, end):
        key = keys[idx]
        entry = data[key]

        # Remove answers from both prompts
        assisted_full = entry.get("assisted", "")
        agentic_full = entry.get("agentic", "")

        assisted_trimmed = remove_answer(assisted_full)
        agentic_trimmed = remove_answer(agentic_full)

        token_entry = token_data.get(key, {})
        agentic_ids = token_entry.get("agentic_ids", [])
        aligned_ids = token_entry.get("assisted_aligned_ids", [])
        aligned_mask = token_entry.get("assisted_aligned_attention_mask", [])
        canonical_pos = token_entry.get("canonical_positions", {}) or {}
        fence_start = canonical_pos.get("fence_start", -1)
        aligned_ok = token_entry.get("aligned_ok", True)

        if fence_start is None or fence_start < 0:
            aligned_ok = False
            fence_start = len(agentic_ids)

        # Remove answer tokens using canonical fence position
        agentic_prefix_ids = agentic_ids[:fence_start]
        assisted_prefix_ids = aligned_ids[:fence_start]
        assisted_prefix_mask = aligned_mask[:fence_start]

        for variant_name, variant_ids, variant_mask in [
            ("agentic", agentic_prefix_ids, [1] * len(agentic_prefix_ids)),
            ("assisted", assisted_prefix_ids, assisted_prefix_mask),
        ]:
            output_path = output_dir / f"{key}_{variant_name}.json"
            if output_path.exists() and not args.overwrite:
                continue

            status = "SUCCESS"
            message = ""
            input_len = 0
            output_len = 0
            t0 = time.time()

            try:
                if variant_ids is None or len(variant_ids) == 0:
                    raise RuntimeError(f"Empty {variant_name} tokens for {key}")

                input_len = len(variant_ids)

                # Prepare tensors
                input_ids_tensor = torch.tensor([variant_ids], device=args.device)
                attention_mask_tensor = torch.tensor([variant_mask], device=args.device)

                # Autoregressive decoding with custom attention_mask
                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=input_ids_tensor,
                        attention_mask=attention_mask_tensor,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=args.do_sample,
                        pad_token_id=pad_token_id,
                    )

                # Decode only the generated part
                generated_ids = outputs[0][input_len:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                output_len = len(generated_ids)

                # Save result
                result = {
                    "key": key,
                    "variant": variant_name,
                    "prompt": agentic_trimmed if variant_name == "agentic" else assisted_trimmed,
                    "generated": generated_text,
                    "input_length": input_len,
                    "output_length": output_len,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "do_sample": args.do_sample,
                    "aligned_ok": aligned_ok if variant_name == "assisted" else True,
                    "fence_start": fence_start,
                }
                output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

            except Exception as exc:
                status = "ERROR"
                message = str(exc)

            finally:
                duration = time.time() - t0
                with log_path.open("a") as f:
                    f.write(
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{key}\t{variant_name}\t{status}\t"
                        f"{input_len}\t{output_len}\t{duration:.2f}\t{message}\n"
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
