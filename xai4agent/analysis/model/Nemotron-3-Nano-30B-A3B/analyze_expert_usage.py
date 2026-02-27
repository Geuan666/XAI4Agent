#!/usr/bin/env python3
import argparse, subprocess, sys
from pathlib import Path

SCRIPT = "/root/autodl-tmp/xai/exp/analyze/analyze_expert_usage.py"
MODEL_PATH = "/root/autodl-tmp/models/Nemotron-3-Nano-30B-A3B"
DATA_PATH = "/root/autodl-tmp/xai/model/Nemotron-3-Nano-30B-A3B/pair_tokens.json"
OUT_DIR = "/root/autodl-tmp/xai/output/Nemotron-3-Nano-30B-A3B"

def main():
    parser = argparse.ArgumentParser(description="Nemotron-3-Nano-30B-A3B expert usage analysis")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    args, rest = parser.parse_known_args()
    cmd = [sys.executable, SCRIPT,
           "--model_path", MODEL_PATH,
           "--data_path", DATA_PATH,
           "--out_dir", OUT_DIR,
           "--device", args.device]
    if args.max_samples is not None:
        cmd += ["--max_samples", str(args.max_samples)]
    cmd += rest
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
