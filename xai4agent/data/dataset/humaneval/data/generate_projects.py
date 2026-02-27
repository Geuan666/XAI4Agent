#!/usr/bin/env python3
"""
从 HumanEval parquet 文件生成测试项目结构

生成的每个项目目录包含：
- main.py: 只有函数签名和文档字符串（函数体为空）
"""

import pandas as pd
import shutil
from pathlib import Path


def generate_projects(
    parquet_path: str,
    output_dir: str,
    clean: bool = True,
):
    """
    生成 HumanEval 测试项目

    Args:
        parquet_path: parquet 文件路径
        output_dir: 输出目录
        clean: 是否先清空输出目录
    """
    # 读取数据
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Found {len(df)} tasks")

    # 清空输出目录
    output_path = Path(output_dir)
    if clean and output_path.exists():
        print(f"Cleaning {output_dir}...")
        shutil.rmtree(output_path)

    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成每个项目
    for idx, row in df.iterrows():
        # 提取 task_id 中的数字
        task_id = row['task_id']  # 格式: "HumanEval/0"
        number = task_id.split('/')[1]
        project_name = f"humaneval_{number}"
        project_dir = output_path / project_name
        project_dir.mkdir(exist_ok=True)

        # 生成 main.py（只有函数签名和文档字符串）
        main_content = row['prompt'].strip()
        # 确保函数体是空的（添加 pass）
        if not main_content.endswith('pass'):
            # 在文档字符串后添加空函数体
            lines = main_content.split('\n')
            # 找到最后一个 """ 的位置
            doc_end_idx = -1
            for i, line in enumerate(lines):
                if line.strip() == '"""':
                    doc_end_idx = i

            if doc_end_idx >= 0:
                # 在文档字符串后添加缩进和 pass
                indent = '    '
                lines.insert(doc_end_idx + 1, indent + 'pass')

            main_content = '\n'.join(lines)

        (project_dir / 'main.py').write_text(main_content + '\n')

        if (idx + 1) % 20 == 0:
            print(f"Generated {idx + 1}/{len(df)} projects...")

    print(f"\nDone! Generated {len(df)} projects in {output_dir}")
    print(f"Project directory: {output_dir}")


if __name__ == '__main__':
    # 配置
    PARQUET_FILE = '/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet'
    OUTPUT_DIR = '/root/autodl-tmp/xai/exp/real/projects'

    generate_projects(
        parquet_path=PARQUET_FILE,
        output_dir=OUTPUT_DIR,
        clean=True,  # 每次重新生成时清空目录
    )
