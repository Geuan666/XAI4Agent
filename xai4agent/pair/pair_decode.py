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

DEFAULT_PROMPTS = "/root/autodl-tmp/XAI4Agent/experiments/pair/pair_prompts.json"
DEFAULT_TOKENS = "/root/autodl-tmp/XAI4Agent/experiments/pair/pair_tokens.json"
DEFAULT_MODEL = "/root/autodl-tmp/qwen3-8B"
DEFAULT_RUN_ROOT = "/root/autodl-tmp/XAI4Agent/experiments/pair"
LEGACY_PROMPTS = "/root/autodl-tmp/xai/exp/pair/pair_prompts.json"
LEGACY_TOKENS = "/root/autodl-tmp/xai/exp/pair/pair_tokens.json"

MARKER = "#Function definition and docstring:\n"
FALLBACK_MARKER = "Function definition and docstring:\n"
ASSISTANT_TAG = "<|im_start|>assistant"
IM_END = "<|im_end|>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
USER_START = "<|im_start|>user"
SYSTEM_START = "<|im_start|>system"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoregressive decode for paired prompts.")
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None, help="Optional run id; default uses current timestamp.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def make_timestamp_id() -> str:
    return time.strftime("%m-%d-%H.%M.%S")


def find_latest_run_file(run_root: str, filename: str) -> Path | None:
    root = Path(run_root)
    if not root.exists():
        return None
    candidates: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        f = d / filename
        if f.exists():
            candidates.append(f)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def infer_run_dir(prompts_path: Path, run_root: str, run_id: str | None) -> Path:
    if prompts_path.name == "pair_prompts.json":
        return prompts_path.parent
    return Path(run_root) / (run_id or make_timestamp_id())


def resolve_prompts_path(args: argparse.Namespace) -> Path:
    if args.run_id:
        candidate = Path(args.run_root) / args.run_id / "pair_prompts.json"
        if candidate.exists():
            return candidate
    prompts_path = Path(args.prompts)
    if prompts_path.exists():
        return prompts_path
    latest = find_latest_run_file(args.run_root, "pair_prompts.json")
    if latest is not None:
        return latest
    legacy_prompts = Path(LEGACY_PROMPTS)
    if legacy_prompts.exists():
        return legacy_prompts
    raise FileNotFoundError(f"prompts not found: {prompts_path}")


def resolve_tokens_path(args: argparse.Namespace, prompts_path: Path) -> Path:
    if prompts_path.name == "pair_prompts.json":
        candidate = prompts_path.with_name("pair_tokens.json")
        if candidate.exists():
            return candidate
    if args.run_id:
        candidate = Path(args.run_root) / args.run_id / "pair_tokens.json"
        if candidate.exists():
            return candidate
    tokens_path = Path(args.tokens)
    if tokens_path.exists():
        return tokens_path
    latest = find_latest_run_file(args.run_root, "pair_tokens.json")
    if latest is not None:
        return latest
    legacy_tokens = Path(LEGACY_TOKENS)
    if legacy_tokens.exists():
        return legacy_tokens
    raise FileNotFoundError(f"tokens not found: {tokens_path}")


def resolve_runtime_paths(args: argparse.Namespace, prompts_path: Path) -> tuple[Path, Path]:
    run_dir = infer_run_dir(prompts_path, args.run_root, args.run_id)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "decode"
    log_path = Path(args.log) if args.log else run_dir / "pair_decode.tsv"
    return output_dir, log_path


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
    prompts_path = resolve_prompts_path(args)
    tokens_path = resolve_tokens_path(args, prompts_path)
    output_dir, log_path = resolve_runtime_paths(args, prompts_path)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    log_path.write_text("timestamp\tkey\tvariant\tstatus\tinput_len\toutput_len\tduration_s\tmessage\n")
    print(f"[pair_decode] output_dir={output_dir}")
    print(f"[pair_decode] log={log_path}")

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
