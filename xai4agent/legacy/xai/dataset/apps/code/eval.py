#!/usr/bin/env python3
import argparse
import json
import math
import re
import signal
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

DEFAULT_PROJECT_ROOT = "/root/autodl-tmp/xai/dataset/apps/code/projects"
DEFAULT_TEST = "/root/autodl-tmp/xai/dataset/apps/test.jsonl"
DEFAULT_PARQUET = "/root/autodl-tmp/xai/dataset/apps/code/humaneval_like/apps_callbased_test.parquet"
DEFAULT_RESULTS = "/root/autodl-tmp/xai/dataset/apps/code/real_eval_results.json"


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


def load_tests(test_path: Path) -> Dict[str, Tuple[str, List[Any], List[Any]]]:
    mapping: Dict[str, Tuple[str, List[Any], List[Any]]] = {}
    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            io_raw = obj.get("input_output") or ""
            if not io_raw:
                continue
            try:
                io_obj = json.loads(io_raw)
            except Exception:
                continue
            if not isinstance(io_obj, dict) or not io_obj.get("fn_name"):
                continue
            fn_name = io_obj.get("fn_name")
            inputs = io_obj.get("inputs") or []
            outputs = io_obj.get("outputs") or []
            key = f"humaneval_{obj.get('id')}"
            mapping[key] = (fn_name, inputs, outputs)
    return mapping


def run_tests(code: str, fn_name: str, inputs: List[Any], outputs: List[Any]) -> Tuple[bool, str]:
    ns: Dict[str, Any] = {"__name__": "__agentic__"}
    try:
        exec(compile(code, "<agentic>", "exec"), ns)
        if fn_name not in ns:
            return False, f"Function '{fn_name}' not found"
        func = ns[fn_name]
        for inp, exp in zip(inputs, outputs):
            if isinstance(inp, (list, tuple)):
                got = func(*inp)
            else:
                got = func(inp)
            if not values_equal(got, exp):
                return False, f"Mismatch: got={got!r} expected={exp!r}"
        return True, ""
    except TimeoutError:
        return False, "Timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Apps (call-based) from main.py files.")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--tests", default=DEFAULT_TEST)
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
    tests = load_tests(Path(args.tests))

    results: Dict[str, Dict[str, Any]] = {}
    success = 0
    fail = 0

    start = max(args.start, 0)
    end = len(df) if args.limit is None else min(len(df), start + args.limit)

    global run_tests
    run_tests = timeout(args.timeout)(run_tests)

    for idx in range(start, end):
        row = df.iloc[idx]
        task_id_raw = row["task_id"]
        key = f"humaneval_{task_id_raw.split('/')[1]}"
        main_path = project_root / key / "main.py"

        if key not in tests:
            results[key] = {"task_id": task_id_raw, "success": False, "error": "Missing tests"}
            fail += 1
            continue
        if not main_path.exists():
            results[key] = {"task_id": task_id_raw, "success": False, "error": "Missing main.py"}
            fail += 1
            continue

        fn_name, inputs, outputs = tests[key]
        code = main_path.read_text()
        ok, err = run_tests(code, fn_name, inputs, outputs)
        results[key] = {"task_id": task_id_raw, "success": ok, "error": err}
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
