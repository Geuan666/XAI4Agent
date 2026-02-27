#!/usr/bin/env python3
"""
从 HumanEval parquet 文件生成 agentic 项目结构（不生成 test.py）

生成的每个项目目录包含：
- main.py: 只有函数签名和文档字符串（函数体为空）
"""

import argparse
import pandas as pd
import shutil
from pathlib import Path


PARQUET_FILE = "/root/autodl-tmp/XAI4Agent/xai4agent/data/dataset/humaneval/data/test-00000-of-00001.parquet"
OUTPUT_DIR = "/root/autodl-tmp/XAI4Agent/xai4agent/pipelines/agentic/projects"
LEGACY_PARQUET_FILE = "/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet"


def generate_projects(parquet_path: str, output_dir: str, clean: bool = True) -> None:
    """生成 HumanEval agentic 项目（仅 main.py）"""
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Found {len(df)} tasks")

    output_path = Path(output_dir)
    if clean and output_path.exists():
        print(f"Cleaning {output_dir}...")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    for idx, row in df.iterrows():
        task_id = row["task_id"]
        number = task_id.split("/")[1]
        project_name = f"humaneval_{number}"
        project_dir = output_path / project_name
        project_dir.mkdir(exist_ok=True)

        main_content = row["prompt"].strip()
        if not main_content.endswith("pass"):
            lines = main_content.split("\n")
            doc_end_idx = -1
            for i, line in enumerate(lines):
                if line.strip() == '"""':
                    doc_end_idx = i
            if doc_end_idx >= 0:
                lines.insert(doc_end_idx + 1, "    pass")
            main_content = "\n".join(lines)

        (project_dir / "main.py").write_text(main_content + "\n")

        if (idx + 1) % 20 == 0:
            print(f"Generated {idx + 1}/{len(df)} projects...")

    print(f"\nDone! Generated {len(df)} projects in {output_dir}")
    print(f"Project directory: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate agentic project directories from a parquet file.")
    parser.add_argument("--parquet", default=PARQUET_FILE, help="Input parquet file path.")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory for projects.")
    parser.add_argument("--no-clean", action="store_true", help="Do not delete existing output directory.")
    return parser.parse_args()


def resolve_parquet(path: str) -> str:
    candidate = Path(path)
    if candidate.exists():
        return str(candidate)
    legacy = Path(LEGACY_PARQUET_FILE)
    if legacy.exists():
        return str(legacy)
    return path


if __name__ == "__main__":
    args = parse_args()
    generate_projects(resolve_parquet(args.parquet), args.output, clean=not args.no_clean)
