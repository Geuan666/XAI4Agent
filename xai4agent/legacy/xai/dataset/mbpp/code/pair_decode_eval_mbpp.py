#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

DEFAULT_DECODE_DIR = "/root/autodl-tmp/xai/output/mbpp/decode"
DEFAULT_PARQUET = "/root/autodl-tmp/xai/dataset/mbpp/code/mbpp_sanitized_test_humaneval.parquet"
DEFAULT_TESTS = "/root/autodl-tmp/xai/dataset/mbpp/sanitized/test-00000-of-00001.parquet"
DEFAULT_RESULTS = "/root/autodl-tmp/xai/output/mbpp/tables/pair_decode_eval_results.json"


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


def get_function_name_from_prompt(prompt: str) -> str:
    match = re.search(r"def\s+(\w+)\s*\(", prompt)
    if not match:
        raise ValueError("Cannot find function definition in prompt")
    return match.group(1)


def construct_full_code(prompt: str, completion: str) -> str:
    completion_code = normalize_completion(completion)
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


def load_tests(path: Path) -> Dict[str, Tuple[List[str], List[str]]]:
    df = pd.read_parquet(path)
    mapping: Dict[str, Tuple[List[str], List[str]]] = {}
    for row in df.itertuples():
        task_id = str(row.task_id)
        imports = _to_list(getattr(row, "test_imports", []))
        tests = _to_list(getattr(row, "test_list", []))
        mapping[task_id] = (imports, tests)
    return mapping


def run_tests(prompt: str, completion: str, imports: list, tests: list, variant: str) -> tuple[bool, str]:
    full_code = construct_full_code(prompt, completion)
    func_name = get_function_name_from_prompt(prompt)

    ns: Dict[str, Any] = {
        "__name__": f"__decode_{variant}__",
        "__builtins__": __builtins__,
    }
    try:
        exec(compile(full_code, "<decode>", "exec"), ns)
        if func_name not in ns:
            return False, f"Function '{func_name}' not found"
        if imports:
            exec(compile("\n".join(imports), "<imports>", "exec"), ns)
        for test in tests:
            exec(compile(test, "<test>", "exec"), ns)
        return True, ""
    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decoded completions for MBPP.")
    parser.add_argument("--decode-dir", default=DEFAULT_DECODE_DIR)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--tests", default=DEFAULT_TESTS)
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
    df_prompts = pd.read_parquet(args.parquet)
    tests_map = load_tests(Path(args.tests))

    results: Dict[str, Dict[str, Any]] = {}
    counts = {"agentic": {"pass": 0, "fail": 0}, "assisted": {"pass": 0, "fail": 0}}

    start = max(args.start, 0)
    end = len(df_prompts) if args.limit is None else min(len(df_prompts), start + args.limit)

    for idx in range(start, end):
        row = df_prompts.iloc[idx]
        task_id_raw = row["task_id"]
        task_num = task_id_raw.split("/")[1]
        task_key = f"humaneval_{task_num}"
        prompt = row["prompt"]

        entry_results: Dict[str, Any] = {"task_id": task_id_raw}
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

            if task_num not in tests_map:
                entry_results[variant] = {"success": False, "error": "Missing tests"}
                counts[variant]["fail"] += 1
                continue

            imports, tests = tests_map[task_num]
            success, error_msg = run_tests(prompt, completion, imports, tests, variant)
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
