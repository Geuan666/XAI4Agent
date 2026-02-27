#!/usr/bin/env python3
"""
Compute activation frequency on tool-mode points for each tool prompt set.

For each prompt (last token), compute router logits and top-k experts.
Count how often each (layer, expert) appears in top-k.
For points from tool-mode mask (mask_tool_last), compute frequency per tool.
Render heatmaps for read_file/write_file/run.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "/root/autodl-tmp/models/qwen3-coder-30b"
DEFAULT_DATA = "/root/autodl-tmp/xai/exp/real/real_tool_prompts.json"
DEFAULT_TOOLMODE = "/root/autodl-tmp/xai/output/humaneval/tables/01-21-09.47/expert_top8_sum.npz"
DEFAULT_OUT = "/root/autodl-tmp/xai/output/humaneval"
DEFAULT_READ_NPZ = "/root/autodl-tmp/xai/output/humaneval/tables/01-22-16.46/tool_prompt_read_file_top8.npz"
DEFAULT_WRITE_NPZ = "/root/autodl-tmp/xai/output/humaneval/tables/01-22-17.02/tool_prompt_write_file_top8.npz"
DEFAULT_RUN_NPZ = "/root/autodl-tmp/xai/output/humaneval/tables/01-22-17.12/tool_prompt_run_top8.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Activation rate on tool-mode points (last token).")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--data_path", default=DEFAULT_DATA)
    parser.add_argument("--toolmode_npz", default=DEFAULT_TOOLMODE)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--timestamped", action="store_true", default=True)
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--use-precomputed", action="store_true", default=False)
    parser.add_argument("--read-npz", default=DEFAULT_READ_NPZ)
    parser.add_argument("--write-npz", default=DEFAULT_WRITE_NPZ)
    parser.add_argument("--run-npz", default=DEFAULT_RUN_NPZ)
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


def normalize_router_logits(router_logits: Any) -> List[torch.Tensor]:
    if router_logits is None:
        return []
    if isinstance(router_logits, torch.Tensor):
        if router_logits.dim() == 4:
            layers = [router_logits[i] for i in range(router_logits.shape[0])]
        else:
            layers = [router_logits]
    else:
        layers = list(router_logits)
    normalized: List[torch.Tensor] = []
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


def run_forward(model: AutoModelForCausalLM, input_ids: List[int], attention_mask: List[int], device: str) -> List[torch.Tensor]:
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


def accumulate_last_token_counts(
    router_logits: List[torch.Tensor], position: int, sum_counts: np.ndarray, k: int
) -> None:
    for l, layer_logits in enumerate(router_logits):
        vec = layer_logits[position].float()
        _, top_idx = torch.topk(vec, k=k, dim=-1)
        contrib = torch.zeros(vec.shape[-1], device=vec.device)
        contrib.scatter_add_(0, top_idx, torch.ones_like(top_idx, dtype=torch.float))
        sum_counts[l] += contrib.detach().cpu().numpy()


def save_masked_heatmap(values: np.ndarray, mask: np.ndarray, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    mat = np.where(mask.T, values.T, np.nan)
    mat = np.ma.masked_invalid(mat)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    vmin = float(np.nanmin(values)) if np.isfinite(values).any() else 0.0
    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    plt.figure(figsize=(12, 8))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    plt.colorbar(im, label="mean activation (top-8 softmax)")
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

    with np.load(args.toolmode_npz) as d:
        tool_mask = d["mask_tool_last"].astype(bool)

    tools = ["read_file", "write_file", "run"]
    use_precomputed = args.use_precomputed
    if args.device == "cuda" and not torch.cuda.is_available():
        use_precomputed = True
        print("CUDA not available; falling back to precomputed counts (requires topk_counts).")

    sum_counts: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {t: 0 for t in tools}
    truncated: Dict[str, int] = {t: 0 for t in tools}
    num_layers = 0
    num_experts = 0

    if use_precomputed:
        npz_map = {
            "read_file": args.read_npz,
            "write_file": args.write_npz,
            "run": args.run_npz,
        }
        for tool in tools:
            path = Path(npz_map[tool])
            if not path.exists():
                raise SystemExit(f"Missing precomputed npz for {tool}: {path}")
            with np.load(path) as d:
                sum_counts[tool] = d.get("topk_counts")
                if sum_counts[tool] is None:
                    raise SystemExit(f"Precomputed npz missing topk_counts for {tool}: {path}")
                sum_counts[tool] = sum_counts[tool].astype(np.float64)
                counts[tool] = int(d.get("filtered_rows", 0))
                truncated[tool] = int(d.get("truncated", 0))
                num_layers = int(d.get("num_layers", sum_counts[tool].shape[0]))
                num_experts = int(d.get("num_experts", sum_counts[tool].shape[1]))
        print("Loaded precomputed sums from tool_prompt_*_top8.npz")
    else:
        data = load_json_list(Path(args.data_path))
        if not isinstance(data, list):
            raise SystemExit("Expected a list in data_path.")

        per_tool_rows: Dict[str, List[dict[str, Any]]] = {t: [] for t in tools}
        for row in data:
            tool = row.get("tool")
            if tool in per_tool_rows and row.get("prompt"):
                per_tool_rows[tool].append(row)

        if args.max_samples is not None:
            for t in tools:
                per_tool_rows[t] = per_tool_rows[t][: args.max_samples]

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

        for tool in tools:
            rows = per_tool_rows[tool]
            if not rows:
                continue
            for i, row in enumerate(rows, 1):
                prompt = row["prompt"]
                enc = tokenizer(prompt, add_special_tokens=False)
                input_ids = enc["input_ids"]
                attn = enc.get("attention_mask") or [1] * len(input_ids)
                if max_len and len(input_ids) > max_len:
                    truncated[tool] += 1
                    input_ids = input_ids[-max_len:]
                    attn = attn[-max_len:]
                router_logits = run_forward(model, input_ids, attn, args.device)
                if sum_counts.get(tool) is None:
                    num_layers = len(router_logits)
                    num_experts = router_logits[0].shape[-1]
                    sum_counts[tool] = np.zeros((num_layers, num_experts), dtype=np.float64)
                pos = len(input_ids) - 1
                accumulate_last_token_counts(router_logits, pos, sum_counts[tool], args.top_k)
                counts[tool] += 1
                if i % 50 == 0:
                    print(f"{tool}: processed {i}/{len(rows)}")

        if not sum_counts:
            raise SystemExit("No tool prompts processed.")

    means: Dict[str, float] = {}
    for tool in tools:
        if tool not in sum_counts or counts[tool] == 0:
            continue
        freq = sum_counts[tool] / counts[tool]
        masked_vals = freq[tool_mask]
        means[tool] = float(masked_vals.mean()) if masked_vals.size else 0.0
        fig_path = figs_dir / f"toolmode_points_{tool}_heatmap.png"
        save_masked_heatmap(
            freq,
            tool_mask,
            f"Tool-mode points activation (last token) - {tool}",
            fig_path,
        )
        np.savez(
            tables_dir / f"toolmode_points_{tool}.npz",
            freq=freq,
            sum_counts=sum_counts[tool],
            counts=counts[tool],
            truncated=truncated[tool],
            tool=tool,
            num_layers=num_layers,
            num_experts=num_experts,
        )

    tool_means = [means[t] for t in tools if t in means]
    overall_mean = float(np.mean(tool_means)) if tool_means else 0.0

    print("Per-tool mean activation (tool-mode points):")
    for t in tools:
        if t in means:
            print(f"  {t}: {means[t]:.6f}")
    print(f"Overall mean (avg of 3 tools): {overall_mean:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
