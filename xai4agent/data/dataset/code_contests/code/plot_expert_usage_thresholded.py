#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path

SCRIPT = "/root/autodl-tmp/xai/exp/analyze/plot_expert_usage_thresholded.py"
OUT_DIR = "/root/autodl-tmp/xai/output/code_contests"

def find_latest_npz(base_out: Path):
    tables = base_out / "tables"
    if (tables / "expert_usage_groups.npz").exists():
        return tables / "expert_usage_groups.npz"
    candidates = []
    if tables.exists():
        for sub in tables.iterdir():
            if sub.is_dir():
                f = sub / "expert_usage_groups.npz"
                if f.exists():
                    candidates.append(f)
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]
    return None

def main():
    parser = argparse.ArgumentParser(description="code_contests thresholded expert usage plots")
    parser.add_argument("--npz", default=None)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--min_threshold", type=float, default=0.005)
    parser.add_argument("--groups", default="canonical,canonical_nonws,question,user_prompt")
    parser.add_argument("--no-timestamp", action="store_true")
    args, rest = parser.parse_known_args()

    base_out = Path(OUT_DIR)
    npz_path = Path(args.npz) if args.npz else find_latest_npz(base_out)
    if npz_path is None:
        raise SystemExit("Could not find expert_usage_groups.npz; run analyze_expert_usage first.")

    cmd = [sys.executable, SCRIPT,
           "--npz", str(npz_path),
           "--out_dir", OUT_DIR,
           "--percentile", str(args.percentile),
           "--min_threshold", str(args.min_threshold),
           "--groups", args.groups]
    if args.no_timestamp:
        cmd.append("--no-timestamp")
    cmd += rest
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
