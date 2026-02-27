#!/usr/bin/env python3
"""
Find intersection points across:
  - tool-mode mask (mask_tool_last from expert_top8_sum.npz)
  - read_file top-8 heatmap (topk matrix)
  - write_file top-8 heatmap (topk matrix)
  - run top-8 heatmap (topk matrix)

Outputs:
  - JSON list of points [{"layer": L, "router": R}, ...]
  - PNG heatmap with intersection points
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


DEFAULT_TOOLMODE = "/root/autodl-tmp/xai/output/humaneval/tables/01-21-09.47/expert_top8_sum.npz"
DEFAULT_READ = "/root/autodl-tmp/xai/output/humaneval/tables/01-22-16.46/tool_prompt_read_file_top8.npz"
DEFAULT_WRITE = "/root/autodl-tmp/xai/output/humaneval/tables/01-22-17.02/tool_prompt_write_file_top8.npz"
DEFAULT_RUN = "/root/autodl-tmp/xai/output/humaneval/tables/01-22-17.12/tool_prompt_run_top8.npz"
DEFAULT_OUT = "/root/autodl-tmp/xai/output/humaneval/figures/routers_expert_top8_sum"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intersect tool-mode with tool prompt top-8 points.")
    parser.add_argument("--toolmode", default=DEFAULT_TOOLMODE)
    parser.add_argument("--read", dest="read_npz", default=DEFAULT_READ)
    parser.add_argument("--write", dest="write_npz", default=DEFAULT_WRITE)
    parser.add_argument("--run", dest="run_npz", default=DEFAULT_RUN)
    parser.add_argument("--out_dir", default=DEFAULT_OUT)
    parser.add_argument("--tag", default="intersection")
    return parser.parse_args()


def load_mask_from_topk(path: Path) -> np.ndarray:
    with np.load(path) as d:
        topk = d["topk"]
    return np.isfinite(topk).astype(np.int32)


def main() -> int:
    args = parse_args()
    toolmode_path = Path(args.toolmode)
    read_path = Path(args.read_npz)
    write_path = Path(args.write_npz)
    run_path = Path(args.run_npz)

    if not toolmode_path.exists():
        raise SystemExit(f"toolmode npz not found: {toolmode_path}")
    for p in (read_path, write_path, run_path):
        if not p.exists():
            raise SystemExit(f"npz not found: {p}")

    with np.load(toolmode_path) as d:
        tool_mask = d["mask_tool_last"]

    read_mask = load_mask_from_topk(read_path)
    write_mask = load_mask_from_topk(write_path)
    run_mask = load_mask_from_topk(run_path)

    if tool_mask.shape != read_mask.shape or tool_mask.shape != write_mask.shape or tool_mask.shape != run_mask.shape:
        raise SystemExit(
            f"shape mismatch: tool={tool_mask.shape}, read={read_mask.shape}, write={write_mask.shape}, run={run_mask.shape}"
        )

    inter = (tool_mask > 0) & (read_mask > 0) & (write_mask > 0) & (run_mask > 0)
    points: List[dict] = []
    layers, routers = np.where(inter)
    for l, r in zip(layers.tolist(), routers.tolist()):
        points.append({"layer": int(l), "router": int(r)})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.tag}_points.json"
    json_path.write_text(json.dumps(points, ensure_ascii=False, indent=2))

    # Render heatmap
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["white", "#d62728"])
    mat = inter.astype(np.float32).T  # routers x layers
    plt.figure(figsize=(12, 8))
    plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1, origin="lower")
    plt.title("Intersection of tool-mode & read/write/run top-8 (last token)")
    plt.xlabel("Layer")
    plt.ylabel("Router")
    plt.tight_layout()
    fig_path = out_dir / f"{args.tag}_heatmap.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Intersection points: {len(points)}")
    print(f"Saved points to: {json_path}")
    print(f"Saved heatmap to: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
