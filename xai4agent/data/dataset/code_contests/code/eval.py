#!/usr/bin/env python3
import argparse
import io
import json
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

DEFAULT_PROJECT_ROOT = "/root/autodl-tmp/xai/dataset/code_contests/code/projects"
DEFAULT_PARQUET = "/root/autodl-tmp/xai/dataset/code_contests/data/test-00000-of-00001-9c49eeff30aacaa8.parquet"
DEFAULT_RESULTS = "/root/autodl-tmp/xai/dataset/code_contests/code/real_eval_results.json"


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


def run_with_io(func, inp: str) -> str:
    data = inp if inp.endswith("\n") else inp + "\n"
    buf = io.BytesIO(data.encode("utf-8"))
    txt = io.TextIOWrapper(buf, encoding="utf-8")
    out = io.StringIO()
    old_in, old_out = sys.stdin, sys.stdout
    sys.stdin, sys.stdout = txt, out
    try:
        func()
    finally:
        sys.stdin, sys.stdout = old_in, old_out
    return out.getvalue()


def run_tests(code: str, inputs: List[str], outputs: List[str]) -> Tuple[bool, str]:
    ns: Dict[str, Any] = {"__name__": "__agentic__"}
    try:
        exec(compile(code, "<agentic>", "exec"), ns)
        if "solve" not in ns:
            return False, "solve not found"
        solve = ns["solve"]
        for inp, exp in zip(inputs, outputs):
            got = run_with_io(solve, inp)
            if (got or "").strip() != (exp or "").strip():
                return False, f"Mismatch: got={got!r} expected={exp!r}"
        return True, ""
    except TimeoutError:
        return False, "Timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CodeContests from main.py files.")
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
        key = f"humaneval_{idx}"
        main_path = project_root / key / "main.py"

        if not main_path.exists():
            results[key] = {"task_id": idx, "success": False, "error": "Missing main.py"}
            fail += 1
            continue

        public_tests = row.get("public_tests") if hasattr(row, "get") else row["public_tests"]
        inputs = _to_list(public_tests.get("input") if isinstance(public_tests, dict) else [])
        outputs = _to_list(public_tests.get("output") if isinstance(public_tests, dict) else [])
        if len(inputs) != len(outputs):
            results[key] = {"task_id": idx, "success": False, "error": "Mismatched tests"}
            fail += 1
            continue

        code = main_path.read_text()
        ok, err = run_tests(code, inputs, outputs)
        results[key] = {"task_id": idx, "success": ok, "error": err}
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
