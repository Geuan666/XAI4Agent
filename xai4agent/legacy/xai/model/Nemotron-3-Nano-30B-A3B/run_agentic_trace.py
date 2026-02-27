#!/usr/bin/env python3
"""Run agentic pipeline against Nemotron vLLM to generate message_trace.

This script sets MODEL/BASE_URL and writes outputs locally:
  - agentic_output.json (contains message_trace)
  - agentic_log.tsv

Usage:
  /root/miniconda3/envs/qwen/bin/python /root/autodl-tmp/xai/model/Nemotron-3-Nano-30B-A3B/run_agentic_trace.py \
    --limit 1 --start 0 --max-steps 16
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8002/v1"
DEFAULT_MODEL_NAME = "nemotron-3-nano-30b-a3b"
DEFAULT_PROJECT_ROOT = "/root/autodl-tmp/xai/exp/agentic/projects"
DEFAULT_RUNNER = "/root/autodl-tmp/xai/exp/agentic/agentic_run.py"
DEFAULT_OUT = "/root/autodl-tmp/xai/model/Nemotron-3-Nano-30B-A3B/agentic_output.json"
DEFAULT_LOG = "/root/autodl-tmp/xai/model/Nemotron-3-Nano-30B-A3B/agentic_log.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run agentic trace with Nemotron vLLM.")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    env = os.environ.copy()
    env["BASE_URL"] = args.base_url
    env["MODEL"] = args.model
    env.setdefault("OPENAI_API_KEY", "dummy")

    cmd = [
        "/root/miniconda3/envs/qwen/bin/python",
        DEFAULT_RUNNER,
        "--project-root",
        args.project_root,
        "--output",
        args.output,
        "--log",
        args.log,
        "--start",
        str(args.start),
        "--max-steps",
        str(args.max_steps),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
