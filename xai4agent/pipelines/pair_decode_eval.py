#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUN_ROOT = "/root/autodl-tmp/XAI4Agent/experiments/pair"
DEFAULT_PARQUET = "/root/autodl-tmp/XAI4Agent/xai4agent/data/dataset/humaneval/data/test-00000-of-00001.parquet"
LEGACY_PARQUET = "/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet"


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

    # Remove trailing fence line (e.g. "```" with optional indentation)
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


def run_test(prompt: str, test_code: str, completion: str, variant: str) -> tuple[bool, str]:
    full_code = construct_full_code(prompt, completion)
    func_name = get_function_name_from_prompt(prompt)

    ns: dict[str, Any] = {
        "__name__": f"__decode_{variant}__",
        "__builtins__": __builtins__,
    }
    try:
        exec(compile(full_code, "<decode>", "exec"), ns)
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
    parser = argparse.ArgumentParser(description="Evaluate decoded completions for HumanEval.")
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default=None, help="Optional run id; default uses current timestamp.")
    parser.add_argument("--decode-dir", default=None)
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--results", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def make_timestamp_id() -> str:
    return time.strftime("%m-%d-%H.%M.%S")


def find_latest_run_dir_with_decode(run_root: str) -> Path | None:
    root = Path(run_root)
    if not root.exists():
        return None
    candidates: list[Path] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        decode_dir = d / "decode"
        if decode_dir.exists() and decode_dir.is_dir():
            candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.decode_dir:
        decode_dir = Path(args.decode_dir)
    else:
        if args.run_id:
            decode_dir = Path(args.run_root) / args.run_id / "decode"
        else:
            latest = find_latest_run_dir_with_decode(args.run_root)
            if latest is not None:
                decode_dir = latest / "decode"
            else:
                run_id = make_timestamp_id()
                decode_dir = Path(args.run_root) / run_id / "decode"
    results_path = Path(args.results) if args.results else decode_dir.parent / "pair_decode_eval_results.json"
    return decode_dir, results_path


def resolve_parquet(path: str) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    legacy = Path(LEGACY_PARQUET)
    if legacy.exists():
        return legacy
    return candidate


def load_decode_entry(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    decode_dir, results_path = resolve_paths(args)
    if not decode_dir.exists():
        raise FileNotFoundError(f"decode_dir not found: {decode_dir}")
    df = pd.read_parquet(resolve_parquet(args.parquet))

    results: dict[str, dict[str, Any]] = {}
    counts = {"agentic": {"pass": 0, "fail": 0}, "assisted": {"pass": 0, "fail": 0}}

    start = max(args.start, 0)
    end = len(df) if args.limit is None else min(len(df), start + args.limit)

    for idx in range(start, end):
        row = df.iloc[idx]
        task_id_raw = row["task_id"]
        task_key = f"humaneval_{task_id_raw.split('/')[1]}"
        prompt = row["prompt"]
        test_code = row["test"]

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

            success, error_msg = run_test(prompt, test_code, completion, variant)
            entry_results[variant] = {"success": success, "error": error_msg}
            if success:
                counts[variant]["pass"] += 1
            else:
                counts[variant]["fail"] += 1

        results[task_key] = entry_results

    total = (counts["agentic"]["pass"] + counts["agentic"]["fail"])
    print(f"Total tests: {total}")
    for variant in ("agentic", "assisted"):
        total_v = counts[variant]["pass"] + counts[variant]["fail"]
        print(f"[{variant}] Passed: {counts[variant]['pass']} Failed: {counts[variant]['fail']}")
        if total_v:
            rate = counts[variant]["pass"] / total_v * 100
            print(f"[{variant}] Success rate: {rate:.2f}%")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps({"counts": counts, "results": results}, ensure_ascii=False, indent=2))
    print(f"[pair_decode_eval] decode_dir={decode_dir}")
    print(f"Results saved to: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
