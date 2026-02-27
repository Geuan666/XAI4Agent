#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from mbpp_utils import (
    build_prompt,
    extract_function_from_code,
    extract_function_name_from_tests,
    extract_signature_block,
    get_first_test,
)


DEFAULT_ROOT = Path("/root/autodl-tmp/xai/dataset/mbpp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a HumanEval-like parquet from MBPP."
    )
    parser.add_argument("--config", choices=["sanitized", "full"], default="sanitized")
    parser.add_argument("--split", default="test", choices=["train", "test", "validation", "prompt"])
    parser.add_argument("--input", default=None, help="Override input parquet path.")
    parser.add_argument("--output", default=None, help="Output parquet path.")
    parser.add_argument("--id-prefix", default="mbpp")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def resolve_input_path(config: str, split: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return DEFAULT_ROOT / config / f"{split}-00000-of-00001.parquet"


def resolve_output_path(config: str, split: str, override: str | None) -> Path:
    if override:
        return Path(override)
    name = f"mbpp_{config}_{split}_humaneval.parquet"
    return DEFAULT_ROOT / "code" / name


def main() -> int:
    args = parse_args()

    input_path = resolve_input_path(args.config, args.split, args.input)
    output_path = resolve_output_path(args.config, args.split, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    if args.limit is not None:
        df = df.head(args.limit)

    records: list[dict] = []
    missing_func = 0
    missing_sig = 0
    used_fallback = 0

    for row in df.itertuples():
        task_num = row.task_id
        task_id = f"{args.id_prefix}/{task_num}"

        prompt_text = row.prompt if args.config == "sanitized" else row.text
        code_text = row.code
        tests = getattr(row, "test_list", [])

        target_name = extract_function_name_from_tests(tests)
        func_code = extract_function_from_code(code_text, target_name)
        if not func_code:
            missing_func += 1
            func_code = code_text

        signature = extract_signature_block(func_code)
        if not signature:
            missing_sig += 1
            signature = "def solution():"

        example = get_first_test(tests)
        prompt = build_prompt(signature, prompt_text, example)

        records.append(
            {
                "task_id": task_id,
                "prompt": prompt,
                "canonical_solution": func_code.rstrip() + "\n",
            }
        )

        if target_name is None:
            used_fallback += 1

    out_df = pd.DataFrame.from_records(records)
    out_df.to_parquet(output_path, index=False)

    print(f"Wrote {len(out_df)} rows to {output_path}")
    print(f"Missing function extraction: {missing_func}")
    print(f"Missing signature: {missing_sig}")
    print(f"Fallback target (no test-derived name): {used_fallback}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

