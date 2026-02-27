#!/usr/bin/env python3
"""Convert CodeContests test split into HumanEval-like parquet schema."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT = "/root/autodl-tmp/xai/dataset/code_contests/data/test-00000-of-00001-9c49eeff30aacaa8.parquet"
DEFAULT_OUTPUT = "/root/autodl-tmp/xai/dataset/code_contests/humaneval_format/test-00000-of-00001.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CodeContests to HumanEval-like parquet.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CodeContests parquet (test split)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HumanEval-like parquet")
    return parser.parse_args()


def sanitize_doc(text: str) -> str:
    # Avoid terminating the triple-quoted docstring.
    return text.replace('"""', '\\"""')


def build_prompt(name: Any, desc: Any) -> str:
    name_str = str(name).strip() if name is not None else ""
    desc_str = str(desc).strip() if desc is not None else ""
    parts = [p for p in (name_str, desc_str) if p]
    doc = "\n\n".join(parts)
    doc = sanitize_doc(doc)

    lines = doc.splitlines() if doc else []
    indented = "\n".join(["    " + line for line in lines]) if lines else ""

    prompt = "def solve():\n    \"\"\"\n"
    if indented:
        prompt += indented + "\n"
    prompt += "    \"\"\"\n    pass\n"
    return prompt


def pick_solution(sol_dict: Any) -> str:
    if isinstance(sol_dict, dict):
        sols = sol_dict.get("solution")
        if sols is not None:
            # numpy arrays and pandas arrays should be iterable
            try:
                for s in list(sols):
                    if isinstance(s, str) and s.strip():
                        return s
            except Exception:
                pass
    return "pass"


def main() -> int:
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(inp)
    records = []
    for idx, row in df.iterrows():
        task_id = f"CodeContests/{idx}"
        prompt = build_prompt(row.get("name"), row.get("description"))
        canonical_solution = pick_solution(row.get("solutions"))
        records.append(
            {
                "task_id": task_id,
                "prompt": prompt,
                "canonical_solution": canonical_solution,
                "test": "",
                "entry_point": "solve",
            }
        )

    out_df = pd.DataFrame.from_records(records, columns=["task_id", "prompt", "canonical_solution", "test", "entry_point"])
    out_df.to_parquet(out, index=False)
    print(f"Wrote {len(out_df)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
