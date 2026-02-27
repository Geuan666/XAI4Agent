#!/usr/bin/env python3
import argparse
import json
import re
import signal
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


DEFAULT_OUTPUT = "/root/autodl-tmp/xai/dataset/mbpp/code/agentic_output.json"
DEFAULT_DATA = "/root/autodl-tmp/xai/dataset/mbpp/sanitized/test-00000-of-00001.parquet"
DEFAULT_PROMPTS = "/root/autodl-tmp/xai/dataset/mbpp/code/mbpp_sanitized_test_humaneval.parquet"
DEFAULT_RESULTS = "/root/autodl-tmp/xai/dataset/mbpp/code/agentic_eval_results.json"


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
            if not m or m.group(1) != target_name:
                pass
            else:
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


def run_tests(prompt: str, completion: str, test_imports: List[Any], test_list: List[Any]) -> tuple[bool, str]:
    m = re.search(r"def\s+(\w+)\s*\(", prompt)
    target = m.group(1) if m else None
    full_code = construct_full_code(prompt, completion, target)
    ns: Dict[str, Any] = {"__name__": "__agentic__"}
    try:
        exec(compile(full_code, "<agentic>", "exec"), ns)
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
    parser = argparse.ArgumentParser(description="Evaluate MBPP agentic outputs.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
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
    df_tests = pd.read_parquet(args.data)
    df_prompts = pd.read_parquet(args.prompts)

    prompt_map = {}
    for row in df_prompts.itertuples():
        task_id = row.task_id
        key = f"humaneval_{str(task_id).split('/')[1]}"
        prompt_map[key] = row.prompt

    results: Dict[str, Dict[str, Any]] = {}
    success = 0
    fail = 0
    start = max(args.start, 0)
    end = len(df_tests) if args.limit is None else min(len(df_tests), start + args.limit)

    global run_tests
    run_tests = timeout(args.timeout)(run_tests)

    for idx in range(start, end):
        row = df_tests.iloc[idx]
        task_id_raw = f"mbpp/{row.task_id}"
        key = f"humaneval_{row.task_id}"
        prompt = prompt_map.get(key, "")

        entry = completions.get(key, {})
        completion = entry.get("final_response") or entry.get("completion") or ""
        if not completion and entry.get("message_trace"):
            for msg in reversed(entry["message_trace"]):
                if msg.get("type") == "ai" and msg.get("content"):
                    completion = msg.get("content")
                    break

        test_imports = row.get("test_imports") if hasattr(row, "get") else row.test_imports
        test_list = row.get("test_list") if hasattr(row, "get") else row.test_list

        ok, err = run_tests(prompt, completion, test_imports, test_list)
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
