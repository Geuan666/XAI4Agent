#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path

SCRIPT = "/root/autodl-tmp/xai/exp/analyze/analyze_routing.py"
MODEL_PATH = "/root/autodl-tmp/qwen3-8B"
DATA_PATH = "/root/autodl-tmp/xai/dataset/apps/code/pair/pair_tokens.json"
PROMPTS_PATH = "/root/autodl-tmp/xai/dataset/apps/code/pair/pair_prompts.json"
OUT_DIR = "/root/autodl-tmp/xai/output/apps"
LOGITLENS_PER_SAMPLE = f"{OUT_DIR}/hidden_states/figs_all/per_sample"
LOGITLENS_AGGREGATE = f"{OUT_DIR}/hidden_states/figs_all/aggregate/aggregate_metrics.pt"

def main():
    parser = argparse.ArgumentParser(description="apps routing analysis")
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args, rest = parser.parse_known_args()
    cmd = [sys.executable, SCRIPT,
           "--model_path", args.model_path,
           "--data_path", DATA_PATH,
           "--prompts_path", PROMPTS_PATH,
           "--out_dir", OUT_DIR,
           "--logitlens_per_sample_dir", LOGITLENS_PER_SAMPLE,
           "--logitlens_aggregate_pt", LOGITLENS_AGGREGATE,
           "--device", args.device]
    if args.max_samples is not None:
        cmd += ["--max_samples", str(args.max_samples)]
    cmd += rest
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
