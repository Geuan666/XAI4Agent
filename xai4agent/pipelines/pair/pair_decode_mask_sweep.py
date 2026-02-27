#!/usr/bin/env python3
"""Sweep router-mask sizes using pair_decode_mask.py and pair_decode_eval.py.

For each N in sizes (default 15..100 step 5), this script:
1) Updates DEFAULT_ROUTER_MASK_POINTS in pair_decode_mask.py to the top-N points
   from the ordered points JSON.
2) Runs pair_decode_mask.py to produce decode outputs in a unique directory.
3) Runs pair_decode_eval.py to evaluate those outputs.
4) Records summary stats for agentic/assisted accuracy.

The points JSON is expected to be a list of dicts sorted by descending diff,
with keys: layer, router, diff (and optionally agentic, assisted).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

DEFAULT_POINTS_JSON = "/root/autodl-tmp/XAI4Agent/experiments/router_ablation/points/last_token_agentic_minus_assisted_points.json"
DEFAULT_PAIR_DECODE = "/root/autodl-tmp/XAI4Agent/xai4agent/pipelines/pair_decode_mask.py"
DEFAULT_PAIR_EVAL = "/root/autodl-tmp/XAI4Agent/xai4agent/pipelines/pair_decode_eval.py"
DEFAULT_OUT_BASE = "/root/autodl-tmp/XAI4Agent/experiments/router_ablation/decode"
DEFAULT_LOG_BASE = "/root/autodl-tmp/XAI4Agent/experiments/router_ablation/logs"
DEFAULT_RESULTS_DIR = "/root/autodl-tmp/XAI4Agent/experiments/router_ablation/results"

DEFAULT_SIZES = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]


def default_python() -> str:
    qwen_python = Path("/root/miniconda3/envs/qwen/bin/python")
    if qwen_python.exists():
        return str(qwen_python)
    return sys.executable


def parse_sizes(text: str | None) -> list[int]:
    if not text:
        return DEFAULT_SIZES
    parts = [p.strip() for p in text.split(",") if p.strip()]
    sizes: list[int] = []
    for p in parts:
        sizes.append(int(p))
    return sizes


def make_timestamp_id() -> str:
    return time.strftime("%m-%d-%H.%M.%S")


def load_points(path: Path) -> list[tuple[int, int]]:
    data = json.loads(path.read_text())
    points: list[tuple[int, int]] = []
    for item in data:
        points.append((int(item["layer"]), int(item["router"])))
    return points


def format_mask_block(points: Iterable[tuple[int, int]]) -> list[str]:
    lines = ["DEFAULT_ROUTER_MASK_POINTS = {"]
    for layer, router in points:
        lines.append(f"    ({layer}, {router}),")
    lines.append("}")
    return lines


def update_mask_in_file(path: Path, points: list[tuple[int, int]]) -> None:
    text = path.read_text()
    lines = text.splitlines()

    start = None
    end = None
    for i, line in enumerate(lines):
        if line.strip().startswith("DEFAULT_ROUTER_MASK_POINTS = {"):
            start = i
            break
    if start is None:
        raise RuntimeError("DEFAULT_ROUTER_MASK_POINTS block not found")

    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "}":
            end = j
            break
    if end is None:
        raise RuntimeError("DEFAULT_ROUTER_MASK_POINTS block end not found")

    new_block = format_mask_block(points)
    new_lines = lines[:start] + new_block + lines[end + 1 :]
    path.write_text("\n".join(new_lines) + "\n")


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep router masks for pair decode/eval.")
    parser.add_argument("--points-json", default=DEFAULT_POINTS_JSON)
    parser.add_argument("--pair-decode", default=DEFAULT_PAIR_DECODE)
    parser.add_argument("--pair-eval", default=DEFAULT_PAIR_EVAL)
    parser.add_argument("--out-base", default=DEFAULT_OUT_BASE)
    parser.add_argument("--log-base", default=DEFAULT_LOG_BASE)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-id", default=None, help="Optional run id; default uses current timestamp.")
    parser.add_argument("--sizes", default=None, help="Comma-separated sizes list.")
    parser.add_argument("--start", type=int, default=0, help="Start index for decode.")
    parser.add_argument("--limit", type=int, default=None, help="Limit count for decode.")
    parser.add_argument("--eval-start", type=int, default=None, help="Start index for eval (optional).")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit count for eval (optional).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite decode outputs.")
    parser.add_argument("--mask-mode", choices=["replace_topk", "keep_topk"], default="keep_topk")
    parser.add_argument("--device", default="cuda", help="Device for pair_decode_mask.py (cuda/cpu).")
    parser.add_argument("--cuda-visible-devices", default=None, help="Set CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--fallback-cpu", action="store_true", help="Retry on CPU if CUDA OOM.")
    parser.add_argument("--python", default=default_python(), help="Python executable to use.")
    args = parser.parse_args()

    run_id = args.run_id or make_timestamp_id()
    points_path = Path(args.points_json)
    pair_decode_path = Path(args.pair_decode)
    pair_eval_path = Path(args.pair_eval)
    out_base = Path(args.out_base) / run_id
    log_base = Path(args.log_base) / run_id
    results_dir = Path(args.results_dir) / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    log_base.mkdir(parents=True, exist_ok=True)
    out_base.mkdir(parents=True, exist_ok=True)

    sizes = parse_sizes(args.sizes)
    points = load_points(points_path)

    original_text = pair_decode_path.read_text()
    summary = []

    try:
        for n in sizes:
            if n > len(points):
                n = len(points)
            mask_points = points[:n]
            update_mask_in_file(pair_decode_path, mask_points)

            decode_dir = out_base / f"decode_mask_top{n}"
            log_path = log_base / f"pair_decode_mask_top{n}.tsv"
            results_path = results_dir / f"pair_decode_mask_top{n}_eval_results.json"

            if results_path.exists() and not args.overwrite:
                eval_data = json.loads(results_path.read_text())
                counts = eval_data.get("counts", {})
                summary.append(
                    {
                        "top_n": n,
                        "agentic_pass": counts.get("agentic", {}).get("pass", 0),
                        "agentic_fail": counts.get("agentic", {}).get("fail", 0),
                        "assisted_pass": counts.get("assisted", {}).get("pass", 0),
                        "assisted_fail": counts.get("assisted", {}).get("fail", 0),
                    }
                )
                continue

            env = dict(os.environ)
            env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            if args.cuda_visible_devices is not None:
                env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

            decode_cmd = [
                args.python,
                str(pair_decode_path),
                "--output-dir",
                str(decode_dir),
                "--log",
                str(log_path),
                "--device",
                str(args.device),
                "--mask-mode",
                str(args.mask_mode),
            ]
            if args.start:
                decode_cmd += ["--start", str(args.start)]
            if args.limit is not None:
                decode_cmd += ["--limit", str(args.limit)]
            if args.overwrite:
                decode_cmd.append("--overwrite")

            eval_cmd = [
                args.python,
                str(pair_eval_path),
                "--decode-dir",
                str(decode_dir),
                "--results",
                str(results_path),
            ]
            if args.eval_start is not None:
                eval_cmd += ["--start", str(args.eval_start)]
            if args.eval_limit is not None:
                eval_cmd += ["--limit", str(args.eval_limit)]

            decode_res = run_cmd(decode_cmd, env=env)
            if decode_res.returncode != 0:
                if args.fallback_cpu and "CUDA out of memory" in (decode_res.stderr or ""):
                    cpu_env = dict(env)
                    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
                    decode_cmd_cpu = decode_cmd.copy()
                    # Replace device argument value to cpu
                    for i in range(len(decode_cmd_cpu)):
                        if decode_cmd_cpu[i] == "--device" and i + 1 < len(decode_cmd_cpu):
                            decode_cmd_cpu[i + 1] = "cpu"
                            break
                    decode_res = run_cmd(decode_cmd_cpu, env=cpu_env)
                if decode_res.returncode != 0:
                    raise RuntimeError(
                        f"pair_decode_mask.py failed (top{n})\\nSTDOUT:\\n{decode_res.stdout}\\nSTDERR:\\n{decode_res.stderr}"
                    )

            eval_res = run_cmd(eval_cmd, env=env)
            if eval_res.returncode != 0:
                raise RuntimeError(
                    f"pair_decode_eval.py failed (top{n})\\nSTDOUT:\\n{eval_res.stdout}\\nSTDERR:\\n{eval_res.stderr}"
                )

            eval_data = json.loads(results_path.read_text())
            counts = eval_data.get("counts", {})
            summary.append(
                {
                    "top_n": n,
                    "agentic_pass": counts.get("agentic", {}).get("pass", 0),
                    "agentic_fail": counts.get("agentic", {}).get("fail", 0),
                    "assisted_pass": counts.get("assisted", {}).get("pass", 0),
                    "assisted_fail": counts.get("assisted", {}).get("fail", 0),
                }
            )

    finally:
        pair_decode_path.write_text(original_text)

    summary_path = results_dir / "pair_decode_mask_sweep_summary.json"
    summary_csv = results_dir / "pair_decode_mask_sweep_summary.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    with summary_csv.open("w", encoding="utf-8") as f:
        f.write("top_n,agentic_pass,agentic_fail,assisted_pass,assisted_fail\n")
        for row in summary:
            f.write(
                f"{row['top_n']},{row['agentic_pass']},{row['agentic_fail']},"
                f"{row['assisted_pass']},{row['assisted_fail']}\n"
            )

    print(f"Saved summary: {summary_path}")
    print(f"Saved summary CSV: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
