#!/usr/bin/env python3
"""End-to-end pipeline to build pair_prompts.json and pair_tokens.json for CodeContests test split."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_RAW = "/root/autodl-tmp/xai/dataset/code_contests/data/test-00000-of-00001-9c49eeff30aacaa8.parquet"
DEFAULT_HE = "/root/autodl-tmp/xai/dataset/code_contests/humaneval_format/test-00000-of-00001.parquet"
DEFAULT_PROJECTS = "/root/autodl-tmp/xai/dataset/code_contests/code/projects"
DEFAULT_AGENTIC = "/root/autodl-tmp/xai/dataset/code_contests/code/agentic_output.json"
DEFAULT_ASSISTED = "/root/autodl-tmp/xai/dataset/code_contests/code/assisted_output.json"
DEFAULT_PAIR_PROMPTS = "/root/autodl-tmp/xai/dataset/code_contests/code/pair_prompts.json"
DEFAULT_PAIR_TOKENS = "/root/autodl-tmp/xai/dataset/code_contests/code/pair_tokens.json"

CONVERT_SCRIPT = "/root/autodl-tmp/xai/exp/analyze/convert_code_contests_to_humaneval.py"
GEN_PROJECTS = "/root/autodl-tmp/xai/exp/analyze/generate_agentic_projects.py"
AGENTIC_RUN = "/root/autodl-tmp/xai/dataset/code_contests/code/agentic_run_codecontests.py"
ASSISTED_RUN = "/root/autodl-tmp/xai/dataset/code_contests/code/assisted_run_codecontests.py"
PAIR_BUILD = "/root/autodl-tmp/xai/exp/pair/pair_build.py"
TOKEN_BUILD = "/root/autodl-tmp/xai/exp/pair/token_build.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CodeContests pair_prompts.json and pair_tokens.json")
    parser.add_argument("--python", default=sys.executable, help="Python executable to run sub-steps")
    parser.add_argument("--raw-parquet", default=DEFAULT_RAW)
    parser.add_argument("--parquet", default=DEFAULT_HE, help="HumanEval-like parquet path")
    parser.add_argument("--projects", default=DEFAULT_PROJECTS)
    parser.add_argument("--agentic", default=DEFAULT_AGENTIC)
    parser.add_argument("--assisted", default=DEFAULT_ASSISTED)
    parser.add_argument("--pair-prompts", default=DEFAULT_PAIR_PROMPTS)
    parser.add_argument("--pair-tokens", default=DEFAULT_PAIR_TOKENS)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--skip-projects", action="store_true")
    parser.add_argument("--skip-agentic", action="store_true")
    parser.add_argument("--skip-assisted", action="store_true")
    parser.add_argument("--skip-pair-build", action="store_true")
    parser.add_argument("--skip-token-build", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    he_path = Path(args.parquet)
    if not args.skip_convert:
        if not Path(CONVERT_SCRIPT).exists():
            raise SystemExit(f"convert script not found: {CONVERT_SCRIPT}")
        run([
            args.python,
            CONVERT_SCRIPT,
            "--input",
            args.raw_parquet,
            "--output",
            args.parquet,
        ])
    elif not he_path.exists():
        raise SystemExit(f"parquet not found and --skip-convert set: {he_path}")

    if not args.skip_projects:
        run([
            args.python,
            GEN_PROJECTS,
            "--parquet",
            args.parquet,
            "--output",
            args.projects,
        ])

    if not args.skip_agentic:
        run([
            args.python,
            AGENTIC_RUN,
            "--project-root",
            args.projects,
            "--output",
            args.agentic,
        ])

    if not args.skip_assisted:
        run([
            args.python,
            ASSISTED_RUN,
            "--parquet",
            args.parquet,
            "--output",
            args.assisted,
        ])

    if not args.skip_pair_build:
        run([
            args.python,
            PAIR_BUILD,
            "--parquet",
            args.parquet,
            "--assisted",
            args.assisted,
            "--agentic",
            args.agentic,
            "--agentic-fallback",
            args.agentic,
            "--project-root",
            args.projects,
            "--output",
            args.pair_prompts,
        ])

    if not args.skip_token_build:
        run([
            args.python,
            TOKEN_BUILD,
            "--pairs",
            args.pair_prompts,
            "--output",
            args.pair_tokens,
        ])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
