#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI


EXP_ROOT = Path("/root/autodl-tmp/xai/exp")
DEFAULT_MODEL = "/root/autodl-tmp/models/qwen3-coder-30b"
DEFAULT_SERVER = "/root/autodl-tmp/FastAPI/qwen3coder/server1.py"
DEFAULT_AGENT = "/root/autodl-tmp/xai/exp/agentic/agent.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Failed to load module: {path}")
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MBPP pipeline with exp modules.")
    parser.add_argument("--parquet", required=True, help="MBPP HumanEval-like parquet path.")
    parser.add_argument("--project-root", default="/root/autodl-tmp/xai/dataset/mbpp/code/agentic_projects")
    parser.add_argument("--assisted-output", default="/root/autodl-tmp/xai/dataset/mbpp/code/assisted_output.json")
    parser.add_argument("--agentic-output", default="/root/autodl-tmp/xai/dataset/mbpp/code/agentic_output.json")
    parser.add_argument("--pair-prompts", default="/root/autodl-tmp/xai/dataset/mbpp/code/pair_prompts.json")
    parser.add_argument("--pair-tokens", default="/root/autodl-tmp/xai/dataset/mbpp/code/pair_tokens.json")
    parser.add_argument("--logs-dir", default="/root/autodl-tmp/xai/dataset/mbpp/code/logs")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--agent", default=DEFAULT_AGENT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--clean-projects", action="store_true")
    return parser.parse_args()


def run_assisted(assisted_mod, parquet_path: Path, output_path: Path, log_path: Path, start: int, limit: int | None):
    df = pd.read_parquet(parquet_path)
    rows = list(df.itertuples())
    start_idx = max(start, 0)
    end_idx = len(rows) if limit is None else min(len(rows), start_idx + limit)

    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000/v1")
    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("timestamp\tproject\tstatus\tduration_s\tprompt_chars\tcompletion_chars\n")

    results: dict[str, dict] = {}
    for idx in range(start_idx, end_idx):
        row = rows[idx]
        number = row.task_id.split("/")[1]
        project_name = f"humaneval_{number}"
        context = assisted_mod.build_main_content(row.prompt)

        status = "SUCCESS"
        try:
            result = assisted_mod.run_task(client, context)
        except Exception as exc:
            status = "ERROR"
            result = {
                "error": str(exc),
                "duration_s": 0,
                "prompt_chars": 0,
                "completion_chars": 0,
            }

        result["base_url"] = base_url
        results[project_name] = result
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a") as f:
            f.write(
                f"{timestamp}\t{project_name}\t{status}\t{result.get('duration_s', 0):.2f}\t"
                f"{result.get('prompt_chars', 0)}\t{result.get('completion_chars', 0)}\n"
            )

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))


def run_agentic(agentic_mod, project_root: Path, output_path: Path, log_path: Path, start: int, limit: int | None, max_steps: int):
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    if "BASE_URL" in os.environ and "OPENAI_BASE_URL" not in os.environ:
        os.environ["OPENAI_BASE_URL"] = os.environ["BASE_URL"]
    if "MODEL" in os.environ and "QWEN_MODEL" not in os.environ:
        os.environ["QWEN_MODEL"] = os.environ["MODEL"]

    projects = agentic_mod.list_projects(project_root)
    if not projects:
        raise SystemExit(f"No projects found under {project_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("timestamp\tproject\tstatus\tduration_s\tprompt_chars\tcompletion_chars\ttool_calls\n")

    agent_module = agentic_mod.load_agent_module()

    results: dict[str, dict] = {}
    if output_path.exists():
        try:
            results = json.loads(output_path.read_text())
        except Exception:
            results = {}
    start_idx = max(start, 0)
    end_idx = len(projects) if limit is None else min(len(projects), start_idx + limit)

    for idx in range(start_idx, end_idx):
        project_dir = projects[idx]
        project_name = project_dir.name

        status = "SUCCESS"
        try:
            result = agentic_mod.run_task(project_dir, max_steps, agent_module)
        except Exception as exc:
            status = "ERROR"
            result = {
                "error": str(exc),
                "duration_s": 0,
                "prompt_chars": 0,
                "completion_chars": 0,
                "tool_calls": [],
            }

        results[project_name] = result
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a") as f:
            f.write(
                f"{timestamp}\t{project_name}\t{status}\t{result.get('duration_s', 0):.2f}\t"
                f"{result.get('prompt_chars', 0)}\t{result.get('completion_chars', 0)}\t"
                f"{len(result.get('tool_calls', []))}\n"
            )

        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        tmp_path.replace(output_path)


def run_pair_build(pair_mod, parquet: str, assisted: str, agentic: str, project_root: str, output: str, model: str, server: str, agent: str):
    argv = [
        "pair_build.py",
        "--parquet",
        parquet,
        "--assisted",
        assisted,
        "--agentic",
        agentic,
        "--agentic-fallback",
        agentic,
        "--project-root",
        project_root,
        "--output",
        output,
        "--model",
        model,
        "--server",
        server,
        "--agent",
        agent,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        pair_mod.main()
    finally:
        sys.argv = old_argv


def run_token_build(token_mod, pairs: str, output: str, model: str):
    argv = [
        "token_build.py",
        "--pairs",
        pairs,
        "--output",
        output,
        "--model",
        model,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        token_mod.main()
    finally:
        sys.argv = old_argv


def main() -> int:
    args = parse_args()
    parquet_path = Path(args.parquet)
    project_root = Path(args.project_root)
    assisted_output = Path(args.assisted_output)
    agentic_output = Path(args.agentic_output)
    pair_prompts = Path(args.pair_prompts)
    pair_tokens = Path(args.pair_tokens)
    logs_dir = Path(args.logs_dir)

    analyze_mod = load_module(EXP_ROOT / "analyze" / "generate_agentic_projects.py", "generate_agentic_projects")
    assisted_mod = load_module(EXP_ROOT / "assisted" / "assisted_run.py", "assisted_run")
    agentic_mod = load_module(EXP_ROOT / "agentic" / "agentic_run.py", "agentic_run")
    pair_mod = load_module(EXP_ROOT / "pair" / "pair_build.py", "pair_build")
    token_mod = load_module(EXP_ROOT / "pair" / "token_build.py", "token_build")

    analyze_mod.generate_projects(str(parquet_path), str(project_root), clean=args.clean_projects)

    assisted_log = logs_dir / "assisted_log.tsv"
    run_assisted(
        assisted_mod,
        parquet_path,
        assisted_output,
        assisted_log,
        args.start,
        args.limit,
    )

    agentic_log = logs_dir / "agentic_log.tsv"
    run_agentic(
        agentic_mod,
        project_root,
        agentic_output,
        agentic_log,
        args.start,
        args.limit,
        args.max_steps,
    )

    run_pair_build(
        pair_mod,
        str(parquet_path),
        str(assisted_output),
        str(agentic_output),
        str(project_root),
        str(pair_prompts),
        args.model,
        args.server,
        args.agent,
    )

    run_token_build(token_mod, str(pair_prompts), str(pair_tokens), args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

