#!/usr/bin/env python3
"""
Aggregate router top-8 experts for the last token of tool-call prompts.

Input: real_tool_prompts.json (list of {id, prompt, sample_id, tool})
Filter by tool name (read_file/write_file/run). For each prompt:
  - Run forward with output_router_logits=True
  - Take last token router logits for each layer
  - Top-8 + softmax per layer, scatter into expert vector
  - Sum across samples

Output:
  - Heatmap of per-layer top-8 expert sums
  - NPZ with raw sums + topk matrix + stats
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/root/autodl-tmp/models/qwen3-coder-30b"
DEFAULT_DATA = "/root/autodl-tmp/xai/exp/real/real_tool_prompts.json"
DEFAULT_OUT = "/root/autodl-tmp/xai/output/humaneval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate tool-prompt top-8 router experts (last token).")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tools", action="append", default=None, help="Tool name(s): run/read_file/write_file (repeatable or comma-separated).")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--timestamped", action="store_true", default=True)
    parser.add_argument("--no-timestamp", action="store_true")
    return parser.parse_args()


def make_timestamp() -> str:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz)
    return f"{now.month:02d}-{now.day:02d}-{now.hour:02d}.{now.minute:02d}"


def resolve_out_dirs(base_out: Path, timestamped: bool) -> tuple[Path, Path]:
    ts = make_timestamp() if timestamped else ""
    figs_dir = base_out / "figures" / "routers_expert_top8_sum"
    tables_dir = base_out / "tables"
    if ts:
        figs_dir = figs_dir / ts
        tables_dir = tables_dir / ts
    return figs_dir, tables_dir


def load_json_list(path: Path) -> List[dict[str, Any]]:
    try:
        import orjson

        return orjson.loads(path.read_bytes())
    except Exception:
        return json.loads(path.read_text())


def parse_tools(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for item in values:
        for part in item.split(","):
            name = part.strip()
            if name:
                out.append(name)
    # de-dup preserve order
    seen = set()
    result = []
    for name in out:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def normalize_router_logits(router_logits: Any) -> list[torch.Tensor]:
    if router_logits is None:
        return []
    if isinstance(router_logits, torch.Tensor):
        if router_logits.dim() == 4:
            layers = [router_logits[i] for i in range(router_logits.shape[0])]
        else:
            layers = [router_logits]
    else:
        layers = list(router_logits)
    normalized: list[torch.Tensor] = []
    for layer in layers:
        if not isinstance(layer, torch.Tensor):
            continue
        t = layer
        if t.dim() == 2:
            normalized.append(t)
        elif t.dim() == 3:
            normalized.append(t[0])
        else:
            t = t.squeeze()
            if t.dim() == 2:
                normalized.append(t)
            elif t.dim() == 3:
                normalized.append(t[0])
    return normalized


def run_forward(model: AutoModelForCausalLM, input_ids: list[int], attention_mask: list[int], device: str) -> list[torch.Tensor]:
    ids = torch.tensor([input_ids], device=device)
    mask = torch.tensor([attention_mask], device=device)
    with torch.inference_mode():
        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            output_router_logits=True,
            use_cache=False,
            return_dict=True,
        )
    router_logits = getattr(outputs, "router_logits", None)
    if router_logits is None and isinstance(outputs, dict):
        router_logits = outputs.get("router_logits")
    layers = normalize_router_logits(router_logits)
    if not layers:
        raise RuntimeError("Model did not return router_logits")
    return layers


def accumulate_last_token(router_logits: list[torch.Tensor], position: int, sum_usage: np.ndarray, k: int) -> None:
    for l, layer_logits in enumerate(router_logits):
        vec = layer_logits[position].float()
        top_vals, top_idx = torch.topk(vec, k=k, dim=-1)
        top_probs = F.softmax(top_vals, dim=-1)
        contrib = torch.zeros(vec.shape[-1], device=vec.device)
        contrib.scatter_add_(0, top_idx, top_probs)
        sum_usage[l] += contrib.detach().cpu().numpy()


def topk_values(sum_usage: np.ndarray, k: int) -> np.ndarray:
    vals = np.full_like(sum_usage, np.nan, dtype=np.float32)
    for l in range(sum_usage.shape[0]):
        row = sum_usage[l]
        if row.size == 0:
            continue
        idx = np.argsort(-row)[:k]
        vals[l, idx] = row[idx]
    return vals


def save_topk_heatmap(values: np.ndarray, title: str, path: Path, cmap_name: str = "Blues") -> None:
    import matplotlib.pyplot as plt

    mat = np.ma.masked_invalid(values.T)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")
    vmin = float(np.nanmin(values)) if np.isfinite(values).any() else 0.0
    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    plt.figure(figsize=(12, 8))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    plt.colorbar(im, label="accumulated top-8 weight")
    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel("Router")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> int:
    args = parse_args()
    base_out = Path(args.out_dir)
    use_timestamp = bool(args.timestamped) and not bool(args.no_timestamp)
    figs_dir, tables_dir = resolve_out_dirs(base_out, use_timestamp)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = load_json_list(Path(args.data_path))
    if not isinstance(data, list):
        raise SystemExit("Expected a list in data_path")

    tool_filter = parse_tools(args.tools)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
        trust_remote_code=True,
        local_files_only=True,
    ).to(args.device)
    model.eval()

    max_len = getattr(model.config, "max_position_embeddings", None)
    if not max_len or max_len <= 0:
        max_len = getattr(tokenizer, "model_max_length", None)

    filtered: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        tool = row.get("tool")
        if tool_filter and tool not in tool_filter:
            continue
        if not row.get("prompt"):
            continue
        filtered.append(row)

    if args.max_samples is not None:
        filtered = filtered[: args.max_samples]
    if not filtered:
        raise SystemExit("No prompts after filtering.")

    stats = {
        "total_rows": len(data),
        "filtered_rows": len(filtered),
        "truncated": 0,
        "tool_filter": tool_filter,
    }

    sum_usage = None
    num_layers = 0
    num_experts = 0

    for row in filtered:
        prompt = row["prompt"]
        enc = tokenizer(prompt, add_special_tokens=False)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask") or [1] * len(input_ids)
        if max_len and len(input_ids) > max_len:
            stats["truncated"] += 1
            input_ids = input_ids[-max_len:]
            attn = attn[-max_len:]
        router_logits = run_forward(model, input_ids, attn, args.device)
        if sum_usage is None:
            num_layers = len(router_logits)
            num_experts = router_logits[0].shape[-1]
            sum_usage = np.zeros((num_layers, num_experts), dtype=np.float64)
        pos = len(input_ids) - 1
        accumulate_last_token(router_logits, pos, sum_usage, args.top_k)

    if sum_usage is None:
        raise SystemExit("No prompts processed.")

    topk_vals = topk_values(sum_usage, args.top_k)

    tool_suffix = "all" if not tool_filter else "_".join(tool_filter)
    fig_path = figs_dir / f"tool_prompt_{tool_suffix}_top{args.top_k}.png"
    save_topk_heatmap(
        topk_vals,
        f"Top-{args.top_k} experts per layer (tool prompts: {tool_suffix})",
        fig_path,
        cmap_name="Blues",
    )

    np.savez(
        tables_dir / f"tool_prompt_{tool_suffix}_top{args.top_k}.npz",
        sum_usage=sum_usage,
        topk=topk_vals,
        top_k=args.top_k,
        num_layers=num_layers,
        num_experts=num_experts,
        **stats,
    )

    print(f"Saved heatmap to: {fig_path}")
    print(f"Saved matrices to: {tables_dir / f'tool_prompt_{tool_suffix}_top{args.top_k}.npz'}")
    print(f"Stats: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
