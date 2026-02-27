#!/usr/bin/env python3
"""
HumanEval FIM Completion Script using vLLM API

Usage:
    python fim_completion.py

Requirements:
    - vLLM server running with Qwen3-Coder model
    - OpenAI Python SDK: pip install openai
"""

import json
import os
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI


# ===== 配置 =====
PARQUET_FILE = "/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet"
OUTPUT_DIR = Path("/root/autodl-tmp/xai/dataset/humaneval/assisted")
OUTPUT_FILE = OUTPUT_DIR / "output.json"
LOG_FILE = OUTPUT_DIR / "completion_log.tsv"

# vLLM API 配置
VLLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = os.getenv("QWEN_MODEL", "qwen3-8b")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-dummy")

# 生成参数
MAX_TOKENS = 512
TEMPERATURE = 0.0
TIMEOUT = 300  # 单个请求超时时间（秒）


def create_fim_prompt(prompt_text: str) -> str:
    """
    创建 FIM prompt
    格式: <|fim_prefix|> + prefix_code + <|fim_suffix|> + suffix_code + <|fim_middle|>

    对于 HumanEval:
    - prefix_code: 函数签名 + 文档字符串
    - suffix_code: 空 (生成整个函数体)
    """
    prefix_code = prompt_text
    suffix_code = ""
    return f"<|fim_prefix|>{prefix_code}<|fim_suffix|>{suffix_code}<|fim_middle|>"


def call_vllm_api(client: OpenAI, fim_prompt: str) -> tuple[str, int, float]:
    """
    调用 vLLM API 进行 FIM 补全

    使用 /v1/chat/completions 端点，因为 Qwen3-Coder Instruct 需要使用 chat 格式

    Returns:
        (生成的代码, exit_code, 耗时秒数)
    """
    start_time = time.time()

    messages = [
        {"role": "system", "content": "You are a code completion assistant."},
        {"role": "user", "content": fim_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            timeout=TIMEOUT
        )

        duration = time.time() - start_time
        output = response.choices[0].message.content
        return output, 0, duration

    except Exception as e:
        duration = time.time() - start_time
        print(f"  Error: {e}")
        return "", -1, duration


def init_log_file():
    """初始化日志文件"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text("timestamp\ttask_id\tstatus\tduration\toutput_length\n", encoding="utf-8")


def log_result(timestamp: str, task_id: str, status: str, duration: float, output_length: int):
    """写入日志"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp}\t{task_id}\t{status}\t{duration:.1f}\t{output_length}\n")


def process_dataset(df: pd.DataFrame, client: OpenAI) -> dict:
    """处理数据集并生成补全"""
    results = {}
    total = len(df)
    start_time = time.time()

    print("=" * 70)
    print(f"HumanEval FIM Code Completion - 处理 {total} 个样本")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"vLLM 端点: {VLLM_BASE_URL}")
    print(f"模型: {MODEL_NAME}")
    print("=" * 70)
    print()

    success_count = 0
    fail_count = 0

    for idx, row in df.iterrows():
        task_id = row['task_id']
        # 转换 "HumanEval/0" 为 "humaneval_0"
        result_key = f"humaneval_{task_id.split('/')[1]}"

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{idx+1:3d}/{total}] {task_id}... ", end="", flush=True)

        # 创建 FIM prompt
        fim_prompt = create_fim_prompt(row['prompt'])

        # 调用 vLLM API
        output, exit_code, duration = call_vllm_api(client, fim_prompt)

        # 处理结果
        if output:
            results[result_key] = output
            log_result(timestamp, task_id, "SUCCESS", duration, len(output))
            print(f"✓ ({duration:.1f}s, {len(output)} 字符)")
            success_count += 1
        else:
            results[result_key] = ""
            log_result(timestamp, task_id, "FAIL", duration, 0)
            print(f"✗ ({duration:.1f}s)")
            fail_count += 1

    # 汇总
    total_duration = time.time() - start_time
    avg_duration = total_duration / total

    print()
    print("=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"总样本数:     {total}")
    print(f"成功:         {success_count}")
    print(f"失败:         {fail_count}")
    print(f"成功率:       {success_count * 100 // total}%")
    print(f"总耗时:       {total_duration:.1f}s ({total_duration/60:.1f} 分钟)")
    print(f"平均耗时:     {avg_duration:.1f}s")
    print(f"完成时间:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    return results


def save_results(results: dict, output_path: Path):
    """保存结果到 JSON 文件"""
    print(f"\n保存结果到 {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"结果保存成功")


def main():
    """主函数"""
    # 初始化日志
    init_log_file()

    # 初始化 vLLM 客户端
    print(f"连接到 vLLM: {VLLM_BASE_URL}")
    client = OpenAI(
        base_url=VLLM_BASE_URL,
        api_key=API_KEY
    )

    # 加载数据集
    print(f"加载数据集: {PARQUET_FILE}")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"加载了 {len(df)} 个样本\n")

    # 处理数据集
    results = process_dataset(df, client)

    # 保存结果
    save_results(results, OUTPUT_FILE)

    print(f"\n✓ 完成! 结果保存到: {OUTPUT_FILE}")
    print(f"✓ 日志保存到: {LOG_FILE}")


if __name__ == "__main__":
    main()
