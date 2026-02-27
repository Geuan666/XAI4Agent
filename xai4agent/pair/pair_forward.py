#!/usr/bin/env python3
"""Run forward passes on paired prompts and save hidden states."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TOKENS = "/root/autodl-tmp/XAI4Agent/experiments/pair/pair_tokens.json"
DEFAULT_MODEL = "/root/autodl-tmp/qwen3-8B"
DEFAULT_RUN_ROOT = "/root/autodl-tmp/XAI4Agent/experiments/pair"
LEGACY_TOKENS = "/root/autodl-tmp/xai/exp/pair/pair_tokens.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward paired prompts and save hidden states.")
    parser.add_argument("--tokens", default=DEFAULT_TOKENS)
    parser.add_argument("--tokens-cache", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None, help="Optional run id; default uses current timestamp.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument(
        "--save-mode",
        choices=["full", "canonical", "pooled"],
        default="canonical",
        help="full: all tokens; canonical: canonical non-ws tokens; pooled: mean over canonical tokens",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Cap canonical positions to first N tokens (for faster IO).",
    )
    parser.add_argument(
        "--store-ids",
        action="store_true",
        help="Store input_ids and attention masks in output payload.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Do not read/write cache.")
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


def infer_run_dir(tokens_path: Path, run_root: str, run_id: str | None) -> Path:
    if tokens_path.name == "pair_tokens.json":
        return tokens_path.parent
    return Path(run_root) / (run_id or make_timestamp_id())


def resolve_tokens_path(args: argparse.Namespace) -> Path:
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


def resolve_runtime_paths(
    args: argparse.Namespace,
    tokens_path: Path,
) -> tuple[Path, Path, Path | None]:
    run_dir = infer_run_dir(tokens_path, args.run_root, args.run_id)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "hidden_states"
    log_path = Path(args.log) if args.log else run_dir / "pair_forward.tsv"
    if args.no_cache:
        cache_path = None
    else:
        cache_path = Path(args.tokens_cache) if args.tokens_cache else run_dir / "pair_tokens.pt"
    return output_dir, log_path, cache_path


def load_json(path: Path) -> dict[str, Any]:
    try:
        import orjson

        return orjson.loads(path.read_bytes())
    except Exception:
        return json.loads(path.read_text())


def load_tokens(path: Path, cache_path: Path | None, use_cache: bool) -> dict[str, Any]:
    if use_cache and cache_path and cache_path.exists():
        return torch.load(cache_path, map_location="cpu")
    data = load_json(path)
    if use_cache and cache_path:
        try:
            torch.save(data, cache_path)
        except Exception:
            pass
    return data


def resolve_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bfloat16" else torch.float16


def list_keys(data: dict[str, Any]) -> list[str]:
    keys = list(data.keys())
    try:
        keys.sort(key=lambda x: int(x.split("_")[1]))
    except Exception:
        keys.sort()
    return keys


def forward_hidden(
    model: AutoModelForCausalLM,
    input_ids: list[int],
    attention_mask: list[int] | None,
    device: str,
    save_mode: str,
    positions: list[int] | None,
) -> torch.Tensor:
    ids = torch.tensor([input_ids], device=device)
    mask = None
    if attention_mask is not None:
        mask = torch.tensor([attention_mask], device=device)
    pos_idx = None
    if positions:
        pos_idx = torch.tensor(positions, device=device)
    with torch.inference_mode():
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = outputs.hidden_states or ()
    if not hidden_states:
        raise RuntimeError("Model did not return hidden_states")
    layer_states = hidden_states[1:]
    if save_mode == "full":
        stacked = torch.stack(
            [hs[0].detach().cpu().to(torch.float16) for hs in layer_states],
            dim=0,
        )
        return stacked
    if pos_idx is None or pos_idx.numel() == 0:
        raise RuntimeError("No canonical positions available for save-mode")
    if save_mode == "canonical":
        stacked = torch.stack(
            [hs[0].index_select(0, pos_idx).detach().cpu().to(torch.float16) for hs in layer_states],
            dim=0,
        )
        return stacked
    if save_mode == "pooled":
        stacked = torch.stack(
            [
                hs[0].index_select(0, pos_idx).mean(dim=0).detach().cpu().to(torch.float16)
                for hs in layer_states
            ],
            dim=0,
        )
        return stacked
    raise RuntimeError(f"Unknown save-mode: {save_mode}")


def main() -> int:
    args = parse_args()
    tokens_path = resolve_tokens_path(args)
    output_dir, log_path, cache_path = resolve_runtime_paths(args, tokens_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    data = load_tokens(tokens_path, cache_path, use_cache=not args.no_cache)
    keys = list_keys(data)
    start = max(args.start, 0)
    end = len(keys) if args.limit is None else min(len(keys), start + args.limit)

    dtype = resolve_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
    )
    model = model.to(args.device)
    model.eval()

    log_path.write_text("timestamp\tkey\tstatus\tagentic_len\tassisted_len\tpositions\tmode\tduration_s\tmessage\n")
    print(f"[pair_forward] output_dir={output_dir}")
    print(f"[pair_forward] log={log_path}")
    if cache_path:
        print(f"[pair_forward] cache={cache_path}")

    for idx in range(start, end):
        key = keys[idx]
        entry = data[key]
        output_path = output_dir / f"{key}.pt"
        if output_path.exists() and not args.overwrite:
            continue

        status = "SUCCESS"
        message = ""
        t0 = time.time()
        try:
            if not entry.get("aligned_ok", True):
                raise RuntimeError("aligned_ok=false")

            agentic_ids = entry["agentic_ids"]
            assisted_ids = entry["assisted_aligned_ids"]
            assisted_mask = entry.get("assisted_aligned_attention_mask")
            if assisted_mask is None or len(assisted_mask) != len(assisted_ids):
                raise RuntimeError("missing or invalid assisted attention mask")

            agentic_mask = [1] * len(agentic_ids)
            canonical_positions = entry.get("canonical_non_ws_positions") or []
            if args.max_positions:
                canonical_positions = canonical_positions[: args.max_positions]

            agentic_hidden = forward_hidden(
                model,
                agentic_ids,
                agentic_mask,
                args.device,
                args.save_mode,
                canonical_positions,
            )
            assisted_hidden = forward_hidden(
                model,
                assisted_ids,
                assisted_mask,
                args.device,
                args.save_mode,
                canonical_positions,
            )

            payload = {
                "key": key,
                "agentic": {
                    "hidden_states": agentic_hidden,
                },
                "assisted": {
                    "hidden_states": assisted_hidden,
                },
                "token_meta": {
                    "pad_token_id": entry.get("pad_token_id"),
                    "agentic_length": entry.get("agentic_length"),
                    "assisted_length": entry.get("assisted_length"),
                    "canonical_positions": canonical_positions,
                    "save_mode": args.save_mode,
                    "segment_positions": entry.get("segment_positions"),
                    "tool_positions": entry.get("tool_positions"),
                    "message_positions": entry.get("message_positions"),
                    "canonical_meta": entry.get("canonical_positions"),
                    "canonical_non_ws_tokens": entry.get("canonical_non_ws_tokens"),
                },
            }
            if args.store_ids:
                payload["agentic"]["input_ids"] = agentic_ids
                payload["agentic"]["attention_mask"] = agentic_mask
                payload["assisted"]["input_ids"] = assisted_ids
                payload["assisted"]["attention_mask"] = assisted_mask
            torch.save(payload, output_path)
        except Exception as exc:
            status = "ERROR"
            message = str(exc)
        finally:
            duration = time.time() - t0
            with log_path.open("a") as f:
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{key}\t{status}\t"
                    f"{entry.get('agentic_length', 0)}\t{entry.get('assisted_length', 0)}\t"
                    f"{len(entry.get('canonical_non_ws_positions') or [])}\t{args.save_mode}\t"
                    f"{duration:.2f}\t{message}\n"
                )
            del entry
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
