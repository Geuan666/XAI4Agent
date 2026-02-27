#!/usr/bin/env python3
"""Dataset-specific pair decode for CodeContests."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_PROMPTS = "/root/autodl-tmp/xai/dataset/code_contests/code/pair_prompts.json"
DEFAULT_TOKENS = "/root/autodl-tmp/xai/dataset/code_contests/code/pair_tokens.json"
DEFAULT_MODEL = "/root/autodl-tmp/models/qwen3-coder-30b"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/xai/output/code_contests/decode"
DEFAULT_LOG = "/root/autodl-tmp/xai/dataset/code_contests/code/pair_decode.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoregressive decode for CodeContests paired prompts.")
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
    parser.add_argument("--overwrite", action="store_true")
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


def get_ids(entry: dict[str, Any], role: str) -> list[int]:
    if role == "agentic":
        ids = entry.get("agentic_ids") or entry.get("agentic_input_ids") or entry.get("agentic")
    else:
        ids = (
            entry.get("assisted_aligned_ids")
            or entry.get("assisted_input_ids")
            or entry.get("assisted_ids")
            or entry.get("assisted")
        )
    return list(ids or [])


def get_mask(entry: dict[str, Any]) -> list[int]:
    mask = entry.get("assisted_aligned_attention_mask") or entry.get("attention_mask") or entry.get("assisted_attention_mask")
    return list(mask or [])


def get_fence_start(entry: dict[str, Any], agentic_len: int) -> tuple[int, bool]:
    aligned_ok = entry.get("aligned_ok", True)
    canonical_pos = entry.get("canonical_positions") or {}
    if isinstance(canonical_pos, dict):
        fence_start = canonical_pos.get("fence_start", -1)
    else:
        fence_start = -1
    if fence_start is None or fence_start < 0:
        aligned_ok = False
        fence_start = agentic_len
    return int(fence_start), bool(aligned_ok)


def main() -> int:
    args = parse_args()
    prompts_path = Path(args.prompts)
    tokens_path = Path(args.tokens)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)

    data = load_json(prompts_path)
    token_data = load_json(tokens_path)
    keys = list_keys(data)
    start = max(args.start, 0)
    end = len(keys) if args.limit is None else min(len(keys), start + args.limit)

    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model.to(args.device)
    model.eval()

    log_path.write_text("timestamp\tkey\tvariant\tstatus\tinput_len\toutput_len\tduration_s\tmessage\n")

    for idx in range(start, end):
        key = keys[idx]
        entry = data[key]
        assisted_full = entry.get("assisted", "")
        agentic_full = entry.get("agentic", "")
        assisted_trimmed = remove_answer(assisted_full)
        agentic_trimmed = remove_answer(agentic_full)

        token_entry = token_data.get(key, {})
        agentic_ids = get_ids(token_entry, "agentic")
        assisted_ids = get_ids(token_entry, "assisted")
        assisted_mask = get_mask(token_entry)
        fence_start, aligned_ok = get_fence_start(token_entry, len(agentic_ids))

        agentic_prefix_ids = agentic_ids[:fence_start]
        assisted_prefix_ids = assisted_ids[:fence_start]
        assisted_prefix_mask = assisted_mask[:fence_start]

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
                if not variant_ids:
                    raise RuntimeError(f"Empty {variant_name} tokens for {key}")

                input_len = len(variant_ids)
                input_ids_tensor = torch.tensor([variant_ids], device=args.device)
                attention_mask_tensor = torch.tensor([variant_mask], device=args.device)

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

                generated_ids = outputs[0][input_len:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                output_len = len(generated_ids)

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
                status = "FAIL"
                message = f"{type(exc).__name__}: {exc}"
                result = {
                    "key": key,
                    "variant": variant_name,
                    "prompt": agentic_trimmed if variant_name == "agentic" else assisted_trimmed,
                    "generated": "",
                    "input_length": input_len,
                    "output_length": output_len,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "do_sample": args.do_sample,
                    "aligned_ok": aligned_ok if variant_name == "assisted" else True,
                    "fence_start": fence_start,
                    "error": message,
                }
                output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

            duration = time.time() - t0
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{key}\t{variant_name}\t{status}\t"
                    f"{input_len}\t{output_len}\t{duration:.2f}\t{message}\n"
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
