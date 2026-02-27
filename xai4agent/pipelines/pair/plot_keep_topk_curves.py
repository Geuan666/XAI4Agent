#!/usr/bin/env python3
"""
Plot accuracy-vs-mask-size curves from existing sweep outputs.

This script is intentionally simple and robust:
- If a directory contains pair_decode_mask_sweep_summary.json, use it.
- Otherwise, fall back to parsing pair_decode_mask_top*_eval_results.json files.

It renders:
1) One curve PNG per experiment directory (saved into that directory)
2) A combined figure with two subplots (saved under --out-dir)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_EXP1 = "/root/autodl-tmp/xai/output/humaneval/tables/keep_topk_agentic_minus_assisted_full"
DEFAULT_EXP2 = "/root/autodl-tmp/xai/output/humaneval/tables/keep_topk_agentic_last_full"
DEFAULT_OUT_DIR = "/root/autodl-tmp/xai/output/humaneval/tables"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot keep_topk sweep curves.")
    p.add_argument("--exp1", default=DEFAULT_EXP1, help="Experiment directory 1 (tables dir).")
    p.add_argument("--exp1-label", default="keep_topk (agentic - assisted ordering)")
    p.add_argument("--exp2", default=DEFAULT_EXP2, help="Experiment directory 2 (tables dir).")
    p.add_argument("--exp2-label", default="keep_topk (agentic last-token ordering)")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to save combined figure.")
    return p.parse_args()


def _load_json(path: Path) -> Any:
    try:
        import orjson

        return orjson.loads(path.read_bytes())
    except Exception:
        return json.loads(path.read_text())


def load_curve_from_dir(exp_dir: Path) -> list[dict[str, Any]]:
    summary = exp_dir / "pair_decode_mask_sweep_summary.json"
    if summary.exists():
        data = _load_json(summary)
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected summary format: {summary}")
        rows: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            top_n = int(item.get("top_n"))
            ap = int(item.get("agentic_pass", 0))
            af = int(item.get("agentic_fail", 0))
            bp = int(item.get("assisted_pass", 0))
            bf = int(item.get("assisted_fail", 0))
            at = ap + af
            bt = bp + bf
            rows.append(
                {
                    "top_n": top_n,
                    "agentic_pass": ap,
                    "agentic_total": at,
                    "agentic_acc": ap / at if at else 0.0,
                    "assisted_pass": bp,
                    "assisted_total": bt,
                    "assisted_acc": bp / bt if bt else 0.0,
                    "source": "summary",
                }
            )
        rows.sort(key=lambda x: x["top_n"])
        return rows

    # Fallback: parse eval files
    pat = re.compile(r"pair_decode_mask_top(\d+)_eval_results\.json$")
    rows = []
    for path in exp_dir.glob("pair_decode_mask_top*_eval_results.json"):
        m = pat.match(path.name)
        if not m:
            continue
        top_n = int(m.group(1))
        data = _load_json(path)
        counts = data.get("counts", {})
        ap = int(counts.get("agentic", {}).get("pass", 0))
        af = int(counts.get("agentic", {}).get("fail", 0))
        bp = int(counts.get("assisted", {}).get("pass", 0))
        bf = int(counts.get("assisted", {}).get("fail", 0))
        at = ap + af
        bt = bp + bf
        rows.append(
            {
                "top_n": top_n,
                "agentic_pass": ap,
                "agentic_total": at,
                "agentic_acc": ap / at if at else 0.0,
                "assisted_pass": bp,
                "assisted_total": bt,
                "assisted_acc": bp / bt if bt else 0.0,
                "source": "eval_files",
            }
        )
    rows.sort(key=lambda x: x["top_n"])
    return rows


def plot_one(rows: list[dict[str, Any]], title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise RuntimeError("No rows to plot.")

    xs = [r["top_n"] for r in rows]
    ya = [r["agentic_acc"] * 100 for r in rows]
    yb = [r["assisted_acc"] * 100 for r in rows]

    plt.figure(figsize=(10, 4.8))
    plt.plot(xs, ya, marker="o", linewidth=2, label="agentic")
    plt.plot(xs, yb, marker="o", linewidth=2, label="assisted")
    plt.title(title)
    plt.xlabel("# masked routers (top-N)")
    plt.ylabel("accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_two_subplots(
    rows1: list[dict[str, Any]],
    label1: str,
    rows2: list[dict[str, Any]],
    label2: str,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, rows, label in [
        (axes[0], rows1, label1),
        (axes[1], rows2, label2),
    ]:
        if not rows:
            ax.set_title(label + " (missing)")
            ax.axis("off")
            continue
        xs = [r["top_n"] for r in rows]
        ya = [r["agentic_acc"] * 100 for r in rows]
        yb = [r["assisted_acc"] * 100 for r in rows]
        ax.plot(xs, ya, marker="o", linewidth=2, label="agentic")
        ax.plot(xs, yb, marker="o", linewidth=2, label="assisted")
        ax.set_title(label)
        ax.set_xlabel("# masked routers (top-N)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.legend()

    axes[0].set_ylabel("accuracy (%)")
    fig.suptitle("Accuracy vs. masked routers (keep_topk)", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    exp1 = Path(args.exp1)
    exp2 = Path(args.exp2)

    rows1 = load_curve_from_dir(exp1) if exp1.exists() else []
    rows2 = load_curve_from_dir(exp2) if exp2.exists() else []

    if rows1:
        plot_one(rows1, args.exp1_label, exp1 / "pair_decode_mask_accuracy_curve.png")
        print(f"Saved: {exp1 / 'pair_decode_mask_accuracy_curve.png'}")
    else:
        print(f"WARN: no rows for exp1: {exp1}")

    if rows2:
        plot_one(rows2, args.exp2_label, exp2 / "pair_decode_mask_accuracy_curve.png")
        print(f"Saved: {exp2 / 'pair_decode_mask_accuracy_curve.png'}")
    else:
        print(f"WARN: no rows for exp2: {exp2}")

    out_dir = Path(args.out_dir)
    combined = out_dir / "keep_topk_accuracy_curves.png"
    plot_two_subplots(rows1, args.exp1_label, rows2, args.exp2_label, combined)
    print(f"Saved: {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

