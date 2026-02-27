#!/usr/bin/env python3
import argparse
import json
import math
import re
import signal
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


DEFAULT_OUTPUT = "/root/autodl-tmp/xai/dataset/apps/code/assisted_output.json"
DEFAULT_TEST = "/root/autodl-tmp/xai/dataset/apps/test.jsonl"
DEFAULT_PARQUET = "/root/autodl-tmp/xai/dataset/apps/code/humaneval_like/apps_callbased_test.parquet"
DEFAULT_RESULTS = "/root/autodl-tmp/xai/dataset/apps/code/assisted_eval_results.json"


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


def extract_code_snippet(text: str) -> str:
    cleaned = strip_code_fences(text)
    if cleaned != text:
        return cleaned
    lines = cleaned.splitlines()
    if not lines:
        return ""
    for i, line in enumerate(lines):
        if line.startswith("def ") and line.lstrip().startswith("def "):
            return "\n".join(lines[i:]).rstrip()
    for i, line in enumerate(lines):
        if line.startswith(" " * 4) or line.startswith("\t"):
            return "\n".join(lines[i:]).rstrip()
    return cleaned


def extract_body_with_preamble(code: str, target_name: str | None = None) -> str | None:
    lines = code.splitlines()
    def_idx = None
    for i, line in enumerate(lines):
        if line.startswith("def ") and line.lstrip().startswith("def "):
            def_idx = i
            break
    if def_idx is None:
        return None
    if target_name:
        m = re.match(r"def\s+(\w+)\s*\(", lines[def_idx])
        if not m or m.group(1) != target_name:
            return None
    preamble = lines[:def_idx]
    body: List[str] = []
    def_indent = 0
    for j in range(def_idx + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            body.append("")
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= def_indent:
            break
        body.append(line)
    if not body and not preamble:
        return None
    merged = preamble + ([""] if preamble and body else []) + body
    return "\n".join(merged)


def normalize_completion(completion: str, target_name: str | None = None) -> str:
    cleaned = extract_code_snippet(completion)
    extracted = extract_body_with_preamble(cleaned, target_name)
    if extracted:
        cleaned = extracted
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
        if target_name:
            m = re.match(r"def\s+(\w+)\s*\(", lines[0])
            if m and m.group(1) == target_name:
                lines = lines[1:]
                while lines and not lines[0].strip():
                    lines.pop(0)
                if not lines:
                    return ""
        else:
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines.pop(0)
            if not lines:
                return ""

    lines = [line.replace("\t", " " * 4) for line in lines]
    indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    min_indent = min(indents) if indents else 0
    normalized: List[str] = []
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


def construct_full_code(prompt: str, completion: str, target_name: str | None = None) -> str:
    completion_code = normalize_completion(completion, target_name)
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


def run_tests(prompt: str, completion: str, fn_name: str, inputs: List[Any], outputs: List[Any]) -> Tuple[bool, str]:
    full_code = construct_full_code(prompt, completion, fn_name)
    ns: Dict[str, Any] = {"__name__": "__assisted__"}
    try:
        exec(compile(full_code, "<assisted>", "exec"), ns)
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
    except TimeoutError:
        return False, "Timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate APPS assisted outputs.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--tests", default=DEFAULT_TEST)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.exists():
        raise SystemExit(f"Output not found: {output_path}")

    completions = json.loads(output_path.read_text())
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
        prompt = row["prompt"]

        entry = completions.get(key, {})
        completion = entry.get("completion") or entry.get("completion_raw") or entry.get("final_response") or ""

        if key not in tests:
            results[key] = {"task_id": task_id_raw, "success": False, "error": "Missing tests"}
            fail += 1
            continue

        fn_name, inputs, outputs = tests[key]
        ok, err = run_tests(prompt, completion, fn_name, inputs, outputs)
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
