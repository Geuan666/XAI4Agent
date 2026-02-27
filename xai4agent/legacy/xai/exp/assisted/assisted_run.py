#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI


PARQUET_FILE = "/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet"
DEFAULT_OUTPUT_FILE = "/root/autodl-tmp/xai/exp/assisted/assisted_output.json"
DEFAULT_LOG_FILE = "/root/autodl-tmp/xai/exp/assisted/assisted_log.tsv"

USER_TEMPLATE = (
    "You are a code completion assistant.\n"
    "Task: Complete the function body based only on the function definition and docstring.\n"
    "Rules:\n"
    "- Output only the function body (no signature, no docstring).\n"
    "- Preserve correct indentation (4 spaces).\n"
    "- Use only the Python standard library.\n"
    "- Do not add explanations or extra text.\n"
    "- Assume no extra context beyond what is shown.\n"
     "Function definition and docstring:\n"
    "{context}"
)

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assisted coding run for HumanEval.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--log", default=DEFAULT_LOG_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    return parser.parse_args()


def build_main_content(prompt_text: str) -> str:
    main_content = prompt_text.strip()
    if not main_content.endswith("pass"):
        # Insert a stub body after the docstring to keep the file executable.
        lines = main_content.split("\n")
        doc_end_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == '"""':
                doc_end_idx = i
        if doc_end_idx >= 0:
            lines.insert(doc_end_idx + 1, "    pass")
        main_content = "\n".join(lines)
    return main_content + "\n"


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    patterns = [
        r"```python\s*(.*?)```",
        r"```py\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).rstrip()
    return text.rstrip()


def normalize_completion(completion: str) -> str:
    cleaned = strip_code_fences(completion)
    if not cleaned:
        return cleaned

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

    # Normalize indentation to 4 spaces for reliable insertion.
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
                if line.startswith(" " * 4):
                    normalized.append(line)
                else:
                    normalized.append(" " * 4 + line.lstrip())
            else:
                normalized.append(" " * 4 + line.lstrip())

    return "\n".join(normalized)


def run_task(client: OpenAI, context: str) -> dict:
    user_prompt = USER_TEMPLATE.format(context=context)

    messages = [{"role": "user", "content": user_prompt}]

    model = os.getenv("MODEL") or os.getenv("QWEN_MODEL", "qwen3-coder-30b")
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    duration = time.time() - start

    content = response.choices[0].message.content or ""
    normalized = normalize_completion(content)
    return {
        "prompt": user_prompt,
        "prompt_chars": len(user_prompt),
        "completion_raw": content,
        "completion": normalized,
        "completion_chars": len(normalized),
        "duration_s": duration,
        "model": model,
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

    df = pd.read_parquet(PARQUET_FILE)

    output_path = Path(args.output)
    log_path = Path(args.log)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log_path.write_text("timestamp\tproject\tstatus\tduration_s\tprompt_chars\tcompletion_chars\n")

    base_url = os.getenv("BASE_URL", "http://127.0.0.1:8000/v1")
    client = OpenAI(
        base_url=base_url,
        api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    )

    results: dict[str, dict] = {}
    rows = list(df.itertuples())
    start_idx = max(args.start, 0)
    end_idx = len(rows) if args.limit is None else min(len(rows), start_idx + args.limit)

    for idx in range(start_idx, end_idx):
        row = rows[idx]
        number = row.task_id.split("/")[1]
        project_name = f"humaneval_{number}"
        context = build_main_content(row.prompt)

        status = "SUCCESS"
        try:
            result = run_task(client, context)
        except Exception as exc:
            status = "ERROR"
            result = {
                "error": str(exc),
                "duration_s": 0,
                "prompt_chars": 0,
                "completion_chars": 0,
            }

        result["base_url"] = base_url
        results[project_name] = result
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a") as f:
            f.write(
                f"{timestamp}\t{project_name}\t{status}\t{result.get('duration_s', 0):.2f}\t"
                f"{result.get('prompt_chars', 0)}\t{result.get('completion_chars', 0)}\n"
            )

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
