#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PARQUET = "/root/autodl-tmp/XAI4Agent/xai4agent/data/dataset/humaneval/data/test-00000-of-00001.parquet"
DEFAULT_RUN_ROOT = "/root/autodl-tmp/XAI4Agent/experiments/assisted"
LEGACY_PARQUET = "/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet"


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.rstrip()

    # 处理完整的栅栏包裹 (例如: ```python ... ```)
    patterns = [
        r"```python\s*(.*?)```",
        r"```py\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).rstrip()

    # 移除末尾的代码栅栏 (例如: "    ```" 或 "```")
    # 匹配行首可选空格 + 三反引号
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

    # Normalize indentation to 4 spaces for safe concatenation.
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


def run_test(prompt: str, test_code: str, completion: str) -> tuple[bool, str]:
    full_code = construct_full_code(prompt, completion)
    func_name = get_function_name_from_prompt(prompt)

    ns: dict[str, Any] = {
        "__name__": "__assisted__",
        "__builtins__": __builtins__,
    }
    try:
        exec(compile(full_code, "<assisted>", "exec"), ns)
        if func_name not in ns:
            return False, f"Function '{func_name}' not found"
        candidate = ns[func_name]
        exec(compile(test_code, "<test>", "exec"), ns)
        if "check" not in ns:
            return False, "check not found in test"
        ns["check"](candidate)
        return True, ""
    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate assisted completions for HumanEval.")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None, help="Optional run id; default uses current timestamp.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--results", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def make_timestamp_id() -> str:
    return time.strftime("%m-%d-%H.%M.%S")


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    run_dir = Path(args.run_root) / (args.run_id or make_timestamp_id())
    output_path = Path(args.output) if args.output else run_dir / "assisted_output.json"
    results_path = Path(args.results) if args.results else run_dir / "assisted_eval_results.json"
    return output_path, results_path


def resolve_parquet(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    legacy = Path(LEGACY_PARQUET)
    if legacy.exists():
        return legacy
    return candidate


def main() -> int:
    args = parse_args()
    output_path, results_path = resolve_paths(args)
    if not output_path.exists():
        raise SystemExit(f"Output not found: {output_path}")

    completions = json.loads(output_path.read_text())
    df = pd.read_parquet(resolve_parquet(args.parquet))

    results: dict[str, dict[str, Any]] = {}
    success_count = 0
    failed_count = 0

    start = max(args.start, 0)
    end = len(df) if args.limit is None else min(len(df), start + args.limit)

    for idx in range(start, end):
        row = df.iloc[idx]
        task_id_raw = row["task_id"]
        task_key = f"humaneval_{task_id_raw.split('/')[1]}"
        prompt = row["prompt"]
        test_code = row["test"]

        entry = completions.get(task_key, {})
        # Assisted output has 'completion' field directly (already processed)
        completion = entry.get("completion", "")

        if not completion:
            results[task_key] = {
                "task_id": task_id_raw,
                "success": False,
                "error": "No completion found",
            }
            failed_count += 1
            continue

        success, error_msg = run_test(prompt, test_code, completion)
        results[task_key] = {
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

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Results saved to: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
