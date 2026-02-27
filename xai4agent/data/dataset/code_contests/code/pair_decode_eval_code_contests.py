#!/usr/bin/env python3
import argparse
import io
import json
import re
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

DEFAULT_DECODE_DIR = "/root/autodl-tmp/xai/output/code_contests/decode"
DEFAULT_RAW = "/root/autodl-tmp/xai/dataset/code_contests/data/test-00000-of-00001-9c49eeff30aacaa8.parquet"
DEFAULT_HE = "/root/autodl-tmp/xai/dataset/code_contests/humaneval_format/test-00000-of-00001.parquet"
DEFAULT_RESULTS = "/root/autodl-tmp/xai/output/code_contests/tables/pair_decode_eval_results.json"


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


def get_function_name_from_prompt(prompt: str) -> str:
    match = re.search(r"def\s+(\w+)\s*\(", prompt)
    if not match:
        raise ValueError("Cannot find function definition in prompt")
    return match.group(1)


def construct_full_code(prompt: str, completion: str, target_name: str | None = None) -> str:
    completion_code = normalize_completion(completion, target_name)
    prompt = prompt.rstrip()
    if not prompt.endswith("\n"):
        prompt += "\n"
    if prompt.splitlines() and prompt.splitlines()[-1].strip() in ('"""', "'''"):
        return prompt + completion_code
    return prompt + completion_code


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


def run_tests(prompt: str, completion: str, inputs: List[str], outputs: List[str], variant: str) -> tuple[bool, str]:
    full_code = construct_full_code(prompt, completion, "solve")
    func_name = get_function_name_from_prompt(prompt)

    ns: Dict[str, Any] = {
        "__name__": f"__decode_{variant}__",
        "__builtins__": __builtins__,
    }
    try:
        exec(compile(full_code, "<decode>", "exec"), ns)
        if func_name not in ns:
            raise NameError(f"{func_name} not found")
        func = ns[func_name]
        for inp, exp in zip(inputs, outputs):
            got = run_with_io(func, inp)
            if (got or "").strip() != (exp or "").strip():
                return False, f"Mismatch: got={got!r} expected={exp!r}"
        return True, ""
    except (SyntaxError, IndentationError, NameError):
        code_text = strip_code_fences(completion)
        if not code_text.strip():
            return False, "Empty completion"
        for inp, exp in zip(inputs, outputs):
            ns2: Dict[str, Any] = {"__name__": f"__decode_{variant}__"}
            data_out = io.StringIO()
            data_in = inp if inp.endswith("\n") else inp + "\n"
            buf = io.BytesIO(data_in.encode("utf-8"))
            txt = io.TextIOWrapper(buf, encoding="utf-8")
            old_in, old_out = sys.stdin, sys.stdout
            sys.stdin, sys.stdout = txt, data_out
            try:
                exec(compile(code_text, "<decode>", "exec"), ns2)
                if "solve" in ns2 and data_out.getvalue().strip() == "":
                    ns2["solve"]()
            except Exception as e:
                return False, f"{type(e).__name__}: {e}"
            finally:
                sys.stdin, sys.stdout = old_in, old_out
            got = data_out.getvalue()
            if (got or "").strip() != (exp or "").strip():
                return False, f"Mismatch: got={got!r} expected={exp!r}"
        return True, ""
    except TimeoutError:
        return False, "Timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decoded completions for CodeContests.")
    parser.add_argument("--decode-dir", default=DEFAULT_DECODE_DIR)
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--he-parquet", default=DEFAULT_HE)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=10)
    return parser.parse_args()


def load_decode_entry(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> int:
    args = parse_args()
    decode_dir = Path(args.decode_dir)
    df_raw = pd.read_parquet(args.raw)
    df_prompts = pd.read_parquet(args.he_parquet)

    results: Dict[str, Dict[str, Any]] = {}
    counts = {"agentic": {"pass": 0, "fail": 0}, "assisted": {"pass": 0, "fail": 0}}

    start = max(args.start, 0)
    end = len(df_prompts) if args.limit is None else min(len(df_prompts), start + args.limit)

    for idx in range(start, end):
        row = df_prompts.iloc[idx]
        task_id_raw = row["task_id"]
        task_num = int(task_id_raw.split("/")[1])
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

            if task_num >= len(df_raw):
                entry_results[variant] = {"success": False, "error": "Missing tests"}
                counts[variant]["fail"] += 1
                continue

            tests = df_raw.iloc[task_num]["public_tests"]
            inputs = tests.get("input", []) if isinstance(tests, dict) else []
            outputs = tests.get("output", []) if isinstance(tests, dict) else []

            run = timeout(args.timeout)(run_tests)
            try:
                success, error_msg = run(prompt, completion, list(inputs), list(outputs), variant)
            except TimeoutError:
                success, error_msg = False, "Timeout"

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
