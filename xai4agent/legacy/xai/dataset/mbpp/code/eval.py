#!/usr/bin/env python3
import argparse
import json
import math
import signal
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

DEFAULT_PROJECT_ROOT = "/root/autodl-tmp/xai/dataset/mbpp/code/agentic_projects"
DEFAULT_PARQUET = "/root/autodl-tmp/xai/dataset/mbpp/sanitized/test-00000-of-00001.parquet"
DEFAULT_RESULTS = "/root/autodl-tmp/xai/dataset/mbpp/code/real_eval_results.json"


class TimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeoutError("timeout")


def timeout(seconds: int):
    def wrapper(func):
        def inner(*args, **kwargs):
            if seconds and seconds > 0:
                old = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old)
            return func(*args, **kwargs)

        return inner

    return wrapper


def normalize_value(val: Any) -> Any:
    if isinstance(val, tuple):
        return [normalize_value(v) for v in val]
    if isinstance(val, list):
        return [normalize_value(v) for v in val]
    if isinstance(val, dict):
        return {k: normalize_value(v) for k, v in val.items()}
    return val


def values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, bool) or isinstance(b, bool):
            return a == b
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
    return normalize_value(a) == normalize_value(b)


def _to_list(value) -> list:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except Exception:
        return [value]


def run_tests(code: str, test_imports: List[Any], test_list: List[Any]) -> tuple[bool, str]:
    ns: Dict[str, Any] = {"__name__": "__agentic__"}
    try:
        exec(compile(code, "<agentic>", "exec"), ns)
        for imp in _to_list(test_imports):
            if isinstance(imp, str) and imp.strip():
                exec(compile(imp, "<imports>", "exec"), ns)
        for t in _to_list(test_list):
            if isinstance(t, str) and t.strip():
                exec(compile(t, "<test>", "exec"), ns)
        return True, ""
    except TimeoutError:
        return False, "Timeout"
    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MBPP from main.py files.")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root)
    df = pd.read_parquet(args.parquet)

    results: Dict[str, Dict[str, Any]] = {}
    success = 0
    fail = 0

    start = max(args.start, 0)
    end = len(df) if args.limit is None else min(len(df), start + args.limit)

    global run_tests
    run_tests = timeout(args.timeout)(run_tests)

    for idx in range(start, end):
        row = df.iloc[idx]
        task_id = int(row["task_id"])
        key = f"humaneval_{task_id}"
        main_path = project_root / key / "main.py"

        if not main_path.exists():
            results[key] = {"task_id": task_id, "success": False, "error": "Missing main.py"}
            fail += 1
            continue

        code = main_path.read_text()
        ok, err = run_tests(code, row["test_imports"], row["test_list"])
        results[key] = {"task_id": task_id, "success": ok, "error": err}
        if ok:
            success += 1
        else:
            fail += 1

    total = success + fail
    print(f"Total: {total}")
    print(f"Passed: {success}")
    print(f"Failed: {fail}")
    if total:
        print(f"Success rate: {success / total * 100:.2f}%")

    Path(args.results).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Saved: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
