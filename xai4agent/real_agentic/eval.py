#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PROJECT_ROOT = "/root/autodl-tmp/XAI4Agent/xai4agent/real_agentic/projects"
DEFAULT_PARQUET = "/root/autodl-tmp/XAI4Agent/xai4agent/data/dataset/humaneval/data/test-00000-of-00001.parquet"
DEFAULT_RUN_ROOT = "/root/autodl-tmp/XAI4Agent/experiments/real"
LEGACY_PARQUET = "/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HumanEval projects from main.py files.")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None, help="Optional run id; default uses current timestamp.")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--results", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def make_timestamp_id() -> str:
    return time.strftime("%m-%d-%H.%M.%S")


def resolve_results_path(args: argparse.Namespace) -> Path:
    if args.results:
        return Path(args.results)
    run_dir = Path(args.run_root) / (args.run_id or make_timestamp_id())
    return run_dir / "real_eval_results.json"


def resolve_parquet(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    legacy = Path(LEGACY_PARQUET)
    if legacy.exists():
        return legacy
    return candidate


def load_main_code(project_dir: Path) -> str:
    main_path = project_dir / "main.py"
    if not main_path.exists():
        raise FileNotFoundError(f"main.py not found: {main_path}")
    return main_path.read_text()


def run_test(main_code: str, test_code: str, entry_point: str) -> tuple[bool, str]:
    ns: dict[str, Any] = {
        "__name__": "__real_eval__",
        "__builtins__": __builtins__,
    }
    try:
        exec(compile(main_code, "<main>", "exec"), ns)
        if entry_point not in ns:
            return False, f"Function '{entry_point}' not found"
        candidate = ns[entry_point]
        exec(compile(test_code, "<test>", "exec"), ns)
        if "check" not in ns:
            return False, "check not found in test"
        ns["check"](candidate)
        return True, ""
    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root)
    if not project_root.exists():
        raise SystemExit(f"Project root not found: {project_root}")

    df = pd.read_parquet(resolve_parquet(args.parquet))

    results: dict[str, dict[str, Any]] = {}
    success_count = 0
    failed_count = 0

    start = max(args.start, 0)
    end = len(df) if args.limit is None else min(len(df), start + args.limit)

    for idx in range(start, end):
        row = df.iloc[idx]
        task_id_raw = row["task_id"]
        task_number = task_id_raw.split("/")[1]
        project_name = f"humaneval_{task_number}"
        project_dir = project_root / project_name

        try:
            main_code = load_main_code(project_dir)
        except Exception as exc:
            results[project_name] = {
                "task_id": task_id_raw,
                "success": False,
                "error": str(exc),
            }
            failed_count += 1
            continue

        test_code = row["test"]
        entry_point = row["entry_point"]
        success, error_msg = run_test(main_code, test_code, entry_point)
        results[project_name] = {
            "task_id": task_id_raw,
            "success": success,
            "error": error_msg,
        }
        if success:
            success_count += 1
        else:
            failed_count += 1

    total = success_count + failed_count
    print(f"Total tests: {total}")
    print(f"Passed: {success_count}")
    print(f"Failed: {failed_count}")
    if total:
        print(f"Success rate: {success_count / total * 100:.2f}%")

    results_path = resolve_results_path(args)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Results saved to: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
