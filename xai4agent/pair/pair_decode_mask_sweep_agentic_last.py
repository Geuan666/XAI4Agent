#!/usr/bin/env python3
"""Sweep router-mask sizes using agentic last-token activation ordering.

This script:
1) Loads avg_last_agentic (48x128) from expert_usage_matrices.npz
2) Sorts all (layer, router) by activation descending
3) Writes ordered points JSON
4) Runs pair_decode_mask_sweep.py with those points
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_NPZ = "/root/autodl-tmp/xai/output/humaneval/tables/01-19-00.01/expert_usage_matrices.npz"
DEFAULT_POINTS_JSON = "/root/autodl-tmp/xai/output/humaneval/tables/01-19-00.01/avg_last_agentic_points.json"
DEFAULT_PAIR_SWEEP = "/root/autodl-tmp/xai/exp/pair/pair_decode_mask_sweep.py"
DEFAULT_OUT_BASE = "/root/autodl-tmp/xai/output/humaneval/decode_mask_agentic_last"
DEFAULT_LOG_BASE = "/root/autodl-tmp/xai/exp/pair/agentic_last"
DEFAULT_RESULTS_DIR = "/root/autodl-tmp/xai/output/humaneval/tables/agentic_last"

DEFAULT_SIZES = "15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100"


def default_python() -> str:
    qwen_python = Path("/root/miniconda3/envs/qwen/bin/python")
    if qwen_python.exists():
        return str(qwen_python)
    return sys.executable


def build_points(npz_path: Path) -> list[dict]:
    data = np.load(npz_path)
    if "avg_last_agentic" not in data:
        raise KeyError("avg_last_agentic not found in npz")
    mat = data["avg_last_agentic"]
    points = []
    rows, cols = mat.shape
    for layer in range(rows):
        for router in range(cols):
            points.append({
                "layer": int(layer),
                "router": int(router),
                "score": float(mat[layer, router]),
            })
    points.sort(key=lambda x: x["score"], reverse=True)
    return points


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep mask using agentic last-token ordering.")
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    parser.add_argument("--points-json", default=DEFAULT_POINTS_JSON)
    parser.add_argument("--pair-sweep", default=DEFAULT_PAIR_SWEEP)
    parser.add_argument("--out-base", default=DEFAULT_OUT_BASE)
    parser.add_argument("--log-base", default=DEFAULT_LOG_BASE)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--sizes", default=DEFAULT_SIZES)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mask-mode", choices=["replace_topk", "keep_topk"], default="keep_topk")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--python", default=default_python())
    args = parser.parse_args()

    npz_path = Path(args.npz)
    points_path = Path(args.points_json)
    points_path.parent.mkdir(parents=True, exist_ok=True)

    points = build_points(npz_path)
    points_path.write_text(json.dumps(points, ensure_ascii=False, indent=2))

    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    sweep_cmd = [
        args.python,
        args.pair_sweep,
        "--points-json",
        str(points_path),
        "--out-base",
        str(args.out_base),
        "--log-base",
        str(args.log_base),
        "--results-dir",
        str(args.results_dir),
        "--sizes",
        args.sizes,
        "--mask-mode",
        args.mask_mode,
    ]
    if args.start:
        sweep_cmd += ["--start", str(args.start)]
    if args.limit is not None:
        sweep_cmd += ["--limit", str(args.limit)]
    if args.overwrite:
        sweep_cmd.append("--overwrite")
    if args.device:
        sweep_cmd += ["--device", str(args.device)]
    if args.cuda_visible_devices is not None:
        sweep_cmd += ["--cuda-visible-devices", str(args.cuda_visible_devices)]
    if args.fallback_cpu:
        sweep_cmd.append("--fallback-cpu")
    if args.python:
        sweep_cmd += ["--python", str(args.python)]

    run_cmd(sweep_cmd, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
