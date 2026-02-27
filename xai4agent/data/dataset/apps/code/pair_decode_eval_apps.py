#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

DEFAULT_DECODE_DIR = "/root/autodl-tmp/xai/output/apps/decode"
DEFAULT_PARQUET = "/root/autodl-tmp/xai/dataset/apps/code/humaneval_like/apps_callbased_test.parquet"
DEFAULT_TEST = "/root/autodl-tmp/xai/dataset/apps/test.jsonl"
DEFAULT_RESULTS = "/root/autodl-tmp/xai/output/apps/tables/pair_decode_eval_results.json"


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.rstrip()
    patterns = [
        r"```python\s*(.*?)```",
        r"```py\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).rstrip()
    text = re.sub(r"\n\s*```\s*$", "", text)
    return text.rstrip()


def normalize_completion(completion: str) -> str:
    cleaned = strip_code_fences(completion)
    if not cleaned:
        return ""
    lines = cleaned.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""

    if lines[0].lstrip().startswith("def ") and lines[0].startswith("def "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return ""

    lines = [line.replace("\t", " " * 4) for line in lines]
    indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    min_indent = min(indents) if indents else 0

    normalized: list[str] = []
    if min_indent >= 4:
        shift = min_indent - 4
        for line in lines:
            if line.strip():
                normalized.append(line[shift:] if len(line) >= shift else line.lstrip())
            else:
                normalized.append("")
    else:
        has_indented = any(indent >= 4 for indent in indents)
        for line in lines:
            if not line.strip():
                normalized.append("")
            elif has_indented:
                normalized.append(line if line.startswith(" " * 4) else " " * 4 + line.lstrip())
            else:
                normalized.append(" " * 4 + line.lstrip())

    return "\n".join(normalized)


def construct_full_code(prompt: str, completion: str) -> str:
    completion_code = normalize_completion(completion)
    prompt = prompt.rstrip()
    if not prompt.endswith("\n"):
        prompt += "\n"
    if prompt.splitlines() and prompt.splitlines()[-1].strip() in ('"""', "'''"):
        return prompt + completion_code
    return prompt + completion_code


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


def load_tests(test_path: Path) -> Dict[str, Tuple[str, list, list]]:
    mapping: Dict[str, Tuple[str, list, list]] = {}
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


def run_tests(prompt: str, completion: str, fn_name: str, inputs: list, outputs: list) -> tuple[bool, str]:
    full_code = construct_full_code(prompt, completion)
    ns: dict[str, Any] = {"__name__": "__decode_apps__"}
    try:
        exec(compile(full_code, "<decode>", "exec"), ns)
        if fn_name not in ns:
            return False, f"Function '{fn_name}' not found"
        fn = ns[fn_name]
        for inp, exp in zip(inputs, outputs):
            args = inp if isinstance(inp, (list, tuple)) else [inp]
            try:
                result = fn(*args)
            except Exception as e:
                return False, f"RuntimeError: {type(e).__name__}: {e}"
            if not values_equal(result, exp):
                return False, f"Mismatch: got={result!r} expected={exp!r}"
        return True, ""
    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decoded completions for APPS.")
    parser.add_argument("--decode-dir", default=DEFAULT_DECODE_DIR)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--tests", default=DEFAULT_TEST)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_decode_entry(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    decode_dir = Path(args.decode_dir)
    df = pd.read_parquet(args.parquet)
    tests = load_tests(Path(args.tests))

    results: dict[str, dict[str, Any]] = {}
    counts = {"agentic": {"pass": 0, "fail": 0}, "assisted": {"pass": 0, "fail": 0}}

    start = max(args.start, 0)
    end = len(df) if args.limit is None else min(len(df), start + args.limit)

    for idx in range(start, end):
        row = df.iloc[idx]
        task_id_raw = row["task_id"]
        task_key = f"humaneval_{task_id_raw.split('/')[1]}"
        prompt = row["prompt"]

        entry_results: dict[str, Any] = {"task_id": task_id_raw}
        for variant in ("agentic", "assisted"):
            path = decode_dir / f"{task_key}_{variant}.json"
            entry = load_decode_entry(path)
            if not entry:
                entry_results[variant] = {"success": False, "error": "Missing decode output"}
                counts[variant]["fail"] += 1
                continue

            if variant == "assisted" and not entry.get("aligned_ok", True):
                entry_results[variant] = {"success": False, "error": "aligned_ok=false"}
                counts[variant]["fail"] += 1
                continue

            completion = entry.get("generated", "")
            if not completion:
                entry_results[variant] = {"success": False, "error": "No completion found"}
                counts[variant]["fail"] += 1
                continue

            if task_key not in tests:
                entry_results[variant] = {"success": False, "error": "Missing tests"}
                counts[variant]["fail"] += 1
                continue

            fn_name, inputs, outputs = tests[task_key]
            success, error_msg = run_tests(prompt, completion, fn_name, inputs, outputs)
            entry_results[variant] = {"success": success, "error": error_msg}
            if success:
                counts[variant]["pass"] += 1
            else:
                counts[variant]["fail"] += 1

        results[task_key] = entry_results

    total = counts["agentic"]["pass"] + counts["agentic"]["fail"]
    print(f"Total tests: {total}")
    for variant in ("agentic", "assisted"):
        total_v = counts[variant]["pass"] + counts[variant]["fail"]
        print(f"[{variant}] Passed: {counts[variant]['pass']} Failed: {counts[variant]['fail']}")
        if total_v:
            rate = counts[variant]["pass"] / total_v * 100
            print(f"[{variant}] Success rate: {rate:.2f}%")

    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({"counts": counts, "results": results}, ensure_ascii=False, indent=2))
    print(f"Results saved to: {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
