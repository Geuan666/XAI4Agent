#!/usr/bin/env python3
"""Core expert usage + routing analysis for paired (agentic vs assisted) inputs.

Focus:
1) Expert usage matrices (all canonical_nonws tokens + last token)
2) Routing divergence curves (TV/Jaccard/Intersection)
3) Mask-ready point rankings
4) Thresholded diff heatmaps (merged from previous postprocess script)

Outputs (under --out_dir):
- figures/expert_core/*.png
- tables/expert_core_layer_metrics.csv
- tables/expert_core_matrices.npz
- tables/*points.json
- tables/expert_core_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


DEFAULT_MODEL = "/root/autodl-tmp/models/qwen3-coder-30b"
DEFAULT_DATA = "/root/autodl-tmp/xai/exp/pair/pair_tokens.json"
DEFAULT_OUT = "/root/autodl-tmp/xai/output/humaneval"

EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Core expert/routing analysis.")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--data_path", default=DEFAULT_DATA)
    p.add_argument("--out_dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--threshold_percentile", type=float, default=99.0)
    p.add_argument("--threshold_min", type=float, default=0.005)
    p.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    try:
        import orjson

        return orjson.loads(path.read_bytes())
    except Exception:
        return json.loads(path.read_text())


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def list_keys(data: Dict[str, Any]) -> List[str]:
    keys = list(data.keys())
    try:
        keys.sort(key=lambda x: int(x.split("_")[1]))
    except Exception:
        keys.sort()
    return keys


def get_input(entry: Dict[str, Any], role: str) -> Tuple[List[int], List[int]]:
    if role == "agentic":
        ids = entry.get("agentic_input_ids") or entry.get("agentic_ids") or entry.get("agentic")
        mask = entry.get("agentic_attention_mask")
    else:
        ids = (
            entry.get("assisted_aligned_ids")
            or entry.get("assisted_input_ids")
            or entry.get("assisted_ids")
            or entry.get("assisted")
        )
        mask = (
            entry.get("assisted_aligned_attention_mask")
            or entry.get("attention_mask")
            or entry.get("assisted_attention_mask")
        )
    if ids is None:
        raise RuntimeError(f"Missing {role} input ids")
    if mask is None:
        mask = [1] * len(ids)
    if len(mask) != len(ids):
        if len(mask) < len(ids):
            mask = list(mask) + [1] * (len(ids) - len(mask))
        else:
            mask = list(mask)[: len(ids)]
    return list(ids), list(mask)


def filter_positions(pos: List[int], mask: List[int]) -> List[int]:
    return [p for p in pos if 0 <= p < len(mask) and mask[p] == 1]


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

    out: List[torch.Tensor] = []
    for layer in layers:
        if not isinstance(layer, torch.Tensor):
            continue
        t = layer
        if t.dim() == 2:
            out.append(t)
        elif t.dim() == 3:
            out.append(t[0])
        else:
            t = t.squeeze()
            if t.dim() == 2:
                out.append(t)
            elif t.dim() == 3:
                out.append(t[0])
    return out


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


def accumulate_topk_usage(
    router_logits: List[torch.Tensor],
    positions: List[int],
    usage_sum: np.ndarray,
    usage_cnt: np.ndarray,
    top_k: int,
) -> None:
    if not positions:
        return
    pos_idx = torch.tensor(positions, device=router_logits[0].device)
    for l, layer_logits in enumerate(router_logits):
        probs = F.softmax(layer_logits.float(), dim=-1)
        sel = probs.index_select(0, pos_idx)  # [T, E]
        k = min(top_k, sel.shape[-1])
        vals, idx = torch.topk(sel, k=k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True).clamp_min(EPS)
        for t in range(vals.shape[0]):
            usage_sum[l, idx[t].detach().cpu().numpy()] += vals[t].detach().cpu().numpy()
        usage_cnt[l] += sel.shape[0]


def accumulate_last_topk_usage(
    router_logits: List[torch.Tensor],
    pos: int,
    usage_sum: np.ndarray,
    usage_cnt: np.ndarray,
    top_k: int,
) -> None:
    p = torch.tensor([pos], device=router_logits[0].device)
    for l, layer_logits in enumerate(router_logits):
        probs = F.softmax(layer_logits.float(), dim=-1)
        row = probs.index_select(0, p)[0]
        k = min(top_k, row.shape[-1])
        vals, idx = torch.topk(row, k=k, dim=-1)
        vals = vals / vals.sum().clamp_min(EPS)
        usage_sum[l, idx.detach().cpu().numpy()] += vals.detach().cpu().numpy()
        usage_cnt[l] += 1


def compute_routing_layer_metrics(
    router_a: List[torch.Tensor],
    router_b: List[torch.Tensor],
    positions: List[int],
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_layers = len(router_a)
    tv = np.zeros((num_layers,), dtype=np.float64)
    jacc = np.zeros((num_layers,), dtype=np.float64)
    inter = np.zeros((num_layers,), dtype=np.float64)

    if not positions:
        return tv, jacc, inter

    pos = torch.tensor(positions, device=router_a[0].device)

    for l in range(num_layers):
        pa = F.softmax(router_a[l].float(), dim=-1).index_select(0, pos)
        pb = F.softmax(router_b[l].float(), dim=-1).index_select(0, pos)

        k = min(top_k, pa.shape[-1])
        va, ia = torch.topk(pa, k=k, dim=-1)
        vb, ib = torch.topk(pb, k=k, dim=-1)
        va = va / va.sum(dim=-1, keepdim=True).clamp_min(EPS)
        vb = vb / vb.sum(dim=-1, keepdim=True).clamp_min(EPS)

        layer_tv = []
        layer_j = []
        layer_i = []

        for t in range(pa.shape[0]):
            map_a = {int(i): float(v) for i, v in zip(ia[t].tolist(), va[t].tolist())}
            map_b = {int(i): float(v) for i, v in zip(ib[t].tolist(), vb[t].tolist())}
            keys = sorted(set(map_a.keys()) | set(map_b.keys()))
            pva = np.array([map_a.get(x, 0.0) for x in keys], dtype=np.float64)
            pvb = np.array([map_b.get(x, 0.0) for x in keys], dtype=np.float64)
            layer_tv.append(0.5 * np.abs(pva - pvb).sum())

            sa = set(map_a.keys())
            sb = set(map_b.keys())
            inter_sz = len(sa & sb)
            union_sz = max(len(sa | sb), 1)
            layer_i.append(inter_sz)
            layer_j.append(inter_sz / union_sz)

        tv[l] = float(np.mean(layer_tv))
        jacc[l] = float(np.mean(layer_j))
        inter[l] = float(np.mean(layer_i))

    return tv, jacc, inter


def save_heatmap(matrix: np.ndarray, title: str, path: Path, vmin: float | None = None, vmax: float | None = None) -> None:
    plt.figure(figsize=(12, 6))
    if vmin is None:
        vmin = float(np.nanmin(matrix)) if np.isfinite(matrix).any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
    im = plt.imshow(matrix, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="avg top-k weight")
    plt.title(title)
    plt.xlabel("Expert ID")
    plt.ylabel("Layer")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def threshold_map(diff: np.ndarray, pct: float, min_thr: float) -> Tuple[np.ndarray, float]:
    abs_diff = np.abs(diff)
    thr = max(float(min_thr), float(np.percentile(abs_diff, pct)))
    mat = np.zeros_like(diff, dtype=np.int8)
    mat[diff >= thr] = 1
    mat[diff <= -thr] = -1
    return mat, thr


def save_threshold_heatmap(mat: np.ndarray, title: str, path: Path) -> None:
    cmap = ListedColormap(["#2b6cb0", "#ffffff", "#c53030"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    plt.figure(figsize=(12, 6))
    plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    plt.title(title)
    plt.xlabel("Expert ID")
    plt.ylabel("Layer")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def save_curve(y: np.ndarray, title: str, ylabel: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(y)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def save_points_json(values: np.ndarray, out_path: Path, key_name: str = "score") -> None:
    pts = []
    for l in range(values.shape[0]):
        for e in range(values.shape[1]):
            pts.append({"layer": int(l), "router": int(e), key_name: float(values[l, e])})
    pts.sort(key=lambda x: x[key_name], reverse=True)
    out_path.write_text(json.dumps(pts, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    base_out = Path(args.out_dir)
    figs_dir = base_out / "figures" / "expert_core"
    tables_dir = base_out / "tables"
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(Path(args.data_path))
    keys = list_keys(data)
    if args.max_samples is not None:
        keys = keys[: args.max_samples]
    if not keys:
        raise SystemExit("No samples found in data_path")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    # probe shape
    probe = data[keys[0]]
    aid, amask = get_input(probe, "agentic")
    rl = run_forward(model, aid, amask, device)
    num_layers = len(rl)
    num_experts = rl[0].shape[-1]

    usage_all = {
        "agentic": np.zeros((num_layers, num_experts), dtype=np.float64),
        "assisted": np.zeros((num_layers, num_experts), dtype=np.float64),
    }
    usage_all_cnt = {
        "agentic": np.zeros((num_layers,), dtype=np.int64),
        "assisted": np.zeros((num_layers,), dtype=np.int64),
    }

    usage_last = {
        "agentic": np.zeros((num_layers, num_experts), dtype=np.float64),
        "assisted": np.zeros((num_layers, num_experts), dtype=np.float64),
    }
    usage_last_cnt = {
        "agentic": np.zeros((num_layers,), dtype=np.int64),
        "assisted": np.zeros((num_layers,), dtype=np.int64),
    }

    tv_samples = []
    jacc_samples = []
    inter_samples = []

    processed = 0
    skipped = 0

    for key in keys:
        entry = data[key]
        agentic_ids, agentic_mask = get_input(entry, "agentic")
        assisted_ids, assisted_mask = get_input(entry, "assisted")

        canonical = entry.get("canonical_non_ws_positions") or entry.get("canonical_nonws_positions") or []
        if not canonical:
            skipped += 1
            continue

        pos_a = filter_positions(canonical, agentic_mask)
        pos_b = filter_positions(canonical, assisted_mask)
        shared = sorted(set(pos_a) & set(pos_b))
        if not shared:
            skipped += 1
            continue

        router_a = run_forward(model, agentic_ids, agentic_mask, device)
        router_b = run_forward(model, assisted_ids, assisted_mask, device)

        accumulate_topk_usage(router_a, shared, usage_all["agentic"], usage_all_cnt["agentic"], args.top_k)
        accumulate_topk_usage(router_b, shared, usage_all["assisted"], usage_all_cnt["assisted"], args.top_k)

        last_pos = shared[-1]
        accumulate_last_topk_usage(router_a, last_pos, usage_last["agentic"], usage_last_cnt["agentic"], args.top_k)
        accumulate_last_topk_usage(router_b, last_pos, usage_last["assisted"], usage_last_cnt["assisted"], args.top_k)

        tv, jacc, inter = compute_routing_layer_metrics(router_a, router_b, shared, args.top_k)
        tv_samples.append(tv)
        jacc_samples.append(jacc)
        inter_samples.append(inter)

        processed += 1

    if processed == 0:
        raise SystemExit("No valid samples processed")

    avg_all_a = usage_all["agentic"] / np.maximum(usage_all_cnt["agentic"][:, None], 1)
    avg_all_b = usage_all["assisted"] / np.maximum(usage_all_cnt["assisted"][:, None], 1)
    diff_all = avg_all_a - avg_all_b

    avg_last_a = usage_last["agentic"] / np.maximum(usage_last_cnt["agentic"][:, None], 1)
    avg_last_b = usage_last["assisted"] / np.maximum(usage_last_cnt["assisted"][:, None], 1)
    diff_last = avg_last_a - avg_last_b

    tv_mean = np.stack(tv_samples, axis=0).mean(axis=0)
    jacc_mean = np.stack(jacc_samples, axis=0).mean(axis=0)
    inter_mean = np.stack(inter_samples, axis=0).mean(axis=0)

    # heatmaps
    vmax_all = float(np.nanmax([avg_all_a.max(), avg_all_b.max()])) if avg_all_a.size else 1.0
    vmax_last = float(np.nanmax([avg_last_a.max(), avg_last_b.max()])) if avg_last_a.size else 1.0
    vdiff_all = float(np.nanmax(np.abs(diff_all))) if diff_all.size else 1.0
    vdiff_last = float(np.nanmax(np.abs(diff_last))) if diff_last.size else 1.0

    save_heatmap(avg_all_a, f"Expert usage (agentic, all canonical_nonws, top{args.top_k})", figs_dir / "expert_usage_all_agentic.png", 0.0, vmax_all)
    save_heatmap(avg_all_b, f"Expert usage (assisted, all canonical_nonws, top{args.top_k})", figs_dir / "expert_usage_all_assisted.png", 0.0, vmax_all)
    save_heatmap(diff_all, "Expert usage diff (agentic-assisted, all canonical_nonws)", figs_dir / "expert_usage_all_diff.png", -vdiff_all, vdiff_all)

    save_heatmap(avg_last_a, f"Expert usage (agentic, last token, top{args.top_k})", figs_dir / "expert_usage_last_agentic.png", 0.0, vmax_last)
    save_heatmap(avg_last_b, f"Expert usage (assisted, last token, top{args.top_k})", figs_dir / "expert_usage_last_assisted.png", 0.0, vmax_last)
    save_heatmap(diff_last, "Expert usage diff (agentic-assisted, last token)", figs_dir / "expert_usage_last_diff.png", -vdiff_last, vdiff_last)

    # thresholded maps (merged functionality)
    th_all, thr_all = threshold_map(diff_all, args.threshold_percentile, args.threshold_min)
    th_last, thr_last = threshold_map(diff_last, args.threshold_percentile, args.threshold_min)
    save_threshold_heatmap(th_all, f"Thresholded diff (all tokens, thr={thr_all:.4f})", figs_dir / "expert_usage_all_diff_thresholded.png")
    save_threshold_heatmap(th_last, f"Thresholded diff (last token, thr={thr_last:.4f})", figs_dir / "expert_usage_last_diff_thresholded.png")

    # routing curves
    save_curve(tv_mean, f"TV top{args.top_k} per layer", "TV", figs_dir / "routing_tv_curve.png")
    save_curve(jacc_mean, f"Jaccard@top{args.top_k} per layer", "Jaccard", figs_dir / "routing_jaccard_curve.png")
    save_curve(inter_mean, f"Intersection@top{args.top_k} per layer", "Intersection", figs_dir / "routing_intersection_curve.png")

    # points for masking
    save_points_json(diff_last, tables_dir / "last_token_agentic_minus_assisted_points.json", key_name="diff")
    save_points_json(-diff_last, tables_dir / "last_token_assisted_minus_agentic_points.json", key_name="diff")
    save_points_json(avg_last_a, tables_dir / "last_token_agentic_points.json", key_name="score")

    # csv
    layer_csv = tables_dir / "expert_core_layer_metrics.csv"
    with layer_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer",
                "tv_topk",
                "jaccard_topk",
                "intersection_topk",
                "all_agentic_token_count",
                "all_assisted_token_count",
                "last_agentic_token_count",
                "last_assisted_token_count",
            ]
        )
        for l in range(num_layers):
            writer.writerow(
                [
                    int(l),
                    float(tv_mean[l]),
                    float(jacc_mean[l]),
                    float(inter_mean[l]),
                    int(usage_all_cnt["agentic"][l]),
                    int(usage_all_cnt["assisted"][l]),
                    int(usage_last_cnt["agentic"][l]),
                    int(usage_last_cnt["assisted"][l]),
                ]
            )

    npz_path = tables_dir / "expert_core_matrices.npz"
    np.savez(
        npz_path,
        avg_all_agentic=avg_all_a,
        avg_all_assisted=avg_all_b,
        diff_all=diff_all,
        avg_last_agentic=avg_last_a,
        avg_last_assisted=avg_last_b,
        diff_last=diff_last,
        thresholded_all=th_all,
        thresholded_last=th_last,
        tv_mean=tv_mean,
        jaccard_mean=jacc_mean,
        intersection_mean=inter_mean,
        top_k=args.top_k,
        threshold_all=thr_all,
        threshold_last=thr_last,
    )

    summary = {
        "processed_samples": processed,
        "skipped_samples": skipped,
        "top_k": args.top_k,
        "threshold_percentile": args.threshold_percentile,
        "threshold_min": args.threshold_min,
        "threshold_all": thr_all,
        "threshold_last": thr_last,
        "device": device,
        "dtype": str(dtype),
        "max_samples": args.max_samples,
        "inputs": {
            "data_path": str(Path(args.data_path).resolve()),
            "model_path": args.model_path,
        },
        "outputs": {
            "figures_dir": str(figs_dir.resolve()),
            "layer_csv": str(layer_csv.resolve()),
            "matrices_npz": str(npz_path.resolve()),
            "points": [
                str((tables_dir / "last_token_agentic_minus_assisted_points.json").resolve()),
                str((tables_dir / "last_token_assisted_minus_agentic_points.json").resolve()),
                str((tables_dir / "last_token_agentic_points.json").resolve()),
            ],
        },
    }
    summary_path = tables_dir / "expert_core_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"Processed samples: {processed} (skipped={skipped})")
    print(f"Saved figures to: {figs_dir}")
    print(f"Saved layer metrics: {layer_csv}")
    print(f"Saved matrices: {npz_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
