#!/usr/bin/env python3
"""
Assisted Coding Evaluation Toolkit for HumanEval

综合分析工具集 - 分析 output.json 文件，评估 FIM 补全结果
"""
import argparse
import ast
import hashlib
import json
import os
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

ASSISTED_DIR = Path("/root/autodl-tmp/xai/dataset/humaneval/assisted")
DATA_DIR = Path("/root/autodl-tmp/xai/dataset/humaneval/data")
OUTPUT_BASE = Path("/root/autodl-tmp/xai/dataset/humaneval")

PARQUET_FILE = DATA_DIR / "test-00000-of-00001.parquet"
OUTPUT_FILE = ASSISTED_DIR / "output.json"


# ============================================================================
# Part 1: 基础评估 - 评估 FIM 补全结果的正确性
# ============================================================================

def extract_code_from_completion(completion: str) -> str:
    """
    从补全结果中提取纯Python代码
    去除可能的 markdown 代码块标记
    """
    if not completion:
        return ""

    # 移除 markdown 代码块标记
    patterns = [
        r'```python\n(.*?)```',
        r'```py\n(.*?)```',
        r'```\n(.*?)```',
        r'```python(.*?)```',
        r'```(.*?)```',
    ]

    for pattern in patterns:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            code = match.group(1)
            # 只移除右侧空白，保留左侧的缩进
            return code.rstrip('\n\r ')

    # 如果没有匹配到代码块，返回原内容（只移除右侧空白）
    return completion.rstrip('\n\r ')


def construct_function_from_prompt(prompt: str, completion: str) -> str:
    """
    从 prompt 和补全构造完整的函数代码

    Args:
        prompt: 函数签名和文档字符串
        completion: FIM 生成的函数体

    Returns:
        完整的函数代码
    """
    completion_code = extract_code_from_completion(completion)

    # prompt 已经包含了函数签名和文档字符串，只需要添加函数体
    # 确保 prompt 末尾有正确的缩进
    prompt_lines = prompt.split('\n')
    last_line = prompt_lines[-1]

    # 如果最后一行是文档字符串结束，添加函数体
    # 补全代码通常已经有缩进（4空格），直接添加即可
    if last_line.strip() == '"""' or last_line.strip() == "'''":
        full_code = prompt + "\n" + completion_code
    else:
        full_code = prompt + completion_code

    return full_code


def get_function_name_from_prompt(prompt: str) -> str:
    """从 prompt 中提取函数名"""
    match = re.search(r'def\s+(\w+)\s*\(', prompt)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot find function definition in prompt")


def create_module_from_code(code: str, task_id: str) -> Tuple[Optional[object], Optional[str]]:
    """
    从代码字符串创建一个 Python 模块

    Returns:
        (module, error_message) - 成功时返回 (module, None)，失败时返回 (None, error_message)
    """
    module_name = f"humaneval_assisted_{task_id}"

    try:
        # 使用 ast 检查语法
        ast.parse(code)
    except SyntaxError as e:
        return None, f"Syntax error: {e}"

    try:
        # 使用 importlib 创建模块
        import importlib.util
        import tempfile

        # 创建临时文件来加载模块
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name

        spec = importlib.util.spec_from_file_location(module_name, temp_path)
        if spec is None or spec.loader is None:
            return None, f"Cannot create module spec"

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # 清理临时文件
        import os
        try:
            os.unlink(temp_path)
        except:
            pass

        return module, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def run_test_from_string(
    test_code: str,
    candidate_func,
    base_namespace: Optional[dict] = None,
) -> Tuple[bool, str]:
    """
    运行测试代码

    Returns:
        (success, error_message)
    """
    try:
        # 创建测试环境的命名空间（复用候选代码中的辅助函数）
        if base_namespace is not None:
            test_namespace = dict(base_namespace)
        else:
            test_namespace = {}
        test_namespace.setdefault('__name__', '__test__')
        test_namespace.setdefault('__builtins__', __builtins__)

        # 执行测试代码以定义 check 函数
        exec(test_code, test_namespace)

        # 调用 check 函数
        if 'check' in test_namespace:
            test_namespace['check'](candidate_func)
            return True, ""
        else:
            return False, "check function not found in test code"

    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def evaluate_single_task(task_id: str, prompt: str, test_code: str, completion: str) -> Dict[str, Any]:
    """
    评估单个任务

    Returns:
        包含评估结果的字典
    """
    result = {
        'task_id': task_id,
        'success': False,
        'error': '',
        'has_syntax_error': False,
        'has_runtime_error': False,
        'has_test_failure': False,
        'completion_length': len(completion),
        'extracted_code': '',
    }

    # 提取代码
    extracted_code = extract_code_from_completion(completion)
    result['extracted_code'] = extracted_code

    # 构造完整函数
    try:
        full_code = construct_function_from_prompt(prompt, completion)
    except Exception as e:
        result['error'] = f"Failed to construct function: {e}"
        result['has_syntax_error'] = True
        return result

    # 创建模块
    module, error = create_module_from_code(full_code, task_id)
    if error:
        result['error'] = error
        result['has_syntax_error'] = 'Syntax error' in error
        result['has_runtime_error'] = 'Syntax error' not in error
        return result

    # 获取函数
    try:
        func_name = get_function_name_from_prompt(prompt)
        candidate_func = getattr(module, func_name)
    except Exception as e:
        result['error'] = f"Failed to get function: {e}"
        return result

    # 运行测试
    success, error_msg = run_test_from_string(test_code, candidate_func, base_namespace=module.__dict__)
    result['success'] = success
    result['error'] = error_msg
    result['has_test_failure'] = not success and 'Assertion failed' in error_msg
    result['has_runtime_error'] = not success and 'Assertion failed' not in error_msg

    return result


def cmd_eval(args):
    """运行基础评估"""
    # 加载数据
    print(f"Loading data from {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded {len(df)} tasks")

    print(f"Loading completions from {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        completions = json.load(f)
    print(f"Loaded {len(completions)} completions")

    print("\n" + "=" * 80)
    print("Running Assisted Coding Evaluation")
    print("=" * 80)

    results = {}
    success_count = 0
    failed_count = 0

    for idx, row in df.iterrows():
        task_id_raw = row['task_id']  # e.g., "HumanEval/0"
        task_key = f"humaneval_{task_id_raw.split('/')[1]}"  # e.g., "humaneval_0"

        if task_key not in completions:
            print(f"✗ {task_key}: NO COMPLETION")
            results[task_key] = {
                'task_id': task_id_raw,
                'success': False,
                'error': 'No completion found',
                'has_syntax_error': False,
                'has_runtime_error': False,
                'has_test_failure': False,
            }
            failed_count += 1
            continue

        completion = completions[task_key]
        result = evaluate_single_task(
            task_key,
            row['prompt'],
            row['test'],
            completion
        )

        results[task_key] = result

        if result['success']:
            success_count += 1
            print(f"✓ {task_key}: PASSED")
        else:
            failed_count += 1
            error_type = "SYNTAX" if result['has_syntax_error'] else \
                         "RUNTIME" if result['has_runtime_error'] else \
                         "TEST FAIL" if result['has_test_failure'] else "ERROR"
            print(f"✗ {task_key}: FAILED ({error_type}) - {result['error'][:60]}")

    # 打印汇总
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = len(df)
    success_rate = (success_count / total * 100) if total > 0 else 0

    print(f"Total tests:     {total}")
    print(f"Passed:          {success_count}")
    print(f"Failed:          {failed_count}")
    print(f"Success rate:    {success_rate:.2f}%")

    # 统计错误类型
    syntax_errors = sum(1 for r in results.values() if r['has_syntax_error'])
    runtime_errors = sum(1 for r in results.values() if r['has_runtime_error'])
    test_failures = sum(1 for r in results.values() if r['has_test_failure'])
    other_errors = sum(1 for r in results.values() if not r['success'] and not r['has_syntax_error'] and not r['has_runtime_error'] and not r['has_test_failure'])

    print(f"\nError breakdown:")
    print(f"  Syntax errors:  {syntax_errors}")
    print(f"  Runtime errors: {runtime_errors}")
    print(f"  Test failures:  {test_failures}")
    print(f"  Other errors:   {other_errors}")

    # 保存结果
    results_file = OUTPUT_BASE / "assisted_evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total': total,
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': success_rate,
            'syntax_errors': syntax_errors,
            'runtime_errors': runtime_errors,
            'test_failures': test_failures,
            'other_errors': other_errors,
            'results': results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_file}")


# ============================================================================
# Part 2: 分析失败案例
# ============================================================================

def cmd_analyze(args):
    """分析失败案例"""
    results_file = OUTPUT_BASE / "assisted_evaluation_results.json"

    if not results_file.exists():
        print(f"错误: 请先运行 'python assisted_eval.py eval' 生成评估数据")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 100)
    print("Assisted Coding - 失败案例分析")
    print("=" * 100)

    failed_cases = []
    for task_key, result in data['results'].items():
        if not result['success']:
            failed_cases.append({
                'task': task_key,
                'error': result['error'],
                'has_syntax_error': result['has_syntax_error'],
                'has_runtime_error': result['has_runtime_error'],
                'has_test_failure': result['has_test_failure'],
                'completion_length': result.get('completion_length', 0),
                'extracted_code': result.get('extracted_code', ''),
            })

    # 按错误类型分组
    syntax_errors = [c for c in failed_cases if c['has_syntax_error']]
    runtime_errors = [c for c in failed_cases if c['has_runtime_error']]
    test_failures = [c for c in failed_cases if c['has_test_failure']]

    print(f"\n总失败案例: {len(failed_cases)}")
    print(f"  语法错误: {len(syntax_errors)}")
    print(f"  运行时错误: {len(runtime_errors)}")
    print(f"  测试失败: {len(test_failures)}")

    # 显示语法错误详情
    if syntax_errors:
        print("\n" + "=" * 100)
        print("语法错误详情")
        print("=" * 100)
        for case in syntax_errors[:20]:
            print(f"\n• {case['task']}")
            print(f"  错误: {case['error'][:80]}")
            if case['extracted_code']:
                code_preview = case['extracted_code'][:150].replace('\n', ' ')
                print(f"  代码预览: {code_preview}...")

    # 显示运行时错误详情
    if runtime_errors:
        print("\n" + "=" * 100)
        print("运行时错误详情")
        print("=" * 100)
        for case in runtime_errors[:20]:
            print(f"\n• {case['task']}")
            print(f"  错误: {case['error'][:80]}")

    # 显示测试失败详情
    if test_failures:
        print("\n" + "=" * 100)
        print("测试失败详情")
        print("=" * 100)
        for case in test_failures[:20]:
            print(f"\n• {case['task']}")
            print(f"  错误: {case['error'][:100]}")

    # 保存失败案例
    output_file = OUTPUT_BASE / "assisted_failed_cases.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(failed_cases, f, indent=2, ensure_ascii=False)

    print(f"\n失败案例详情已保存到: {output_file}")


# ============================================================================
# Part 3: 生成统计报告
# ============================================================================

def cmd_summary(args):
    """生成综合统计报告"""
    results_file = OUTPUT_BASE / "assisted_evaluation_results.json"

    if not results_file.exists():
        print(f"错误: 请先运行 'python assisted_eval.py eval' 生成评估数据")
        return

    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 100)
    print("                     Assisted Coding HumanEval 评估报告")
    print("=" * 100)

    print(f"""
【总体统计】
  总任务数: {data['total']}
  成功任务: {data['success_count']} ({data['success_rate']:.1f}%)
  失败任务: {data['failed_count']} ({data['failed_count']/data['total']*100:.1f}%)

【错误类型分布】
""")

    error_types = [
        ('语法错误', data['syntax_errors']),
        ('运行时错误', data['runtime_errors']),
        ('测试失败', data['test_failures']),
        ('其他错误', data['other_errors']),
    ]

    for name, count in error_types:
        pct = count / data['total'] * 100 if data['total'] > 0 else 0
        print(f"  {name}: {count:3d} ({pct:.1f}%)")

    # 统计补全长度分布
    print(f"\n【补全长度分布】")
    lengths = []
    for result in data['results'].values():
        if 'completion_length' in result:
            lengths.append(result['completion_length'])

    if lengths:
        length_ranges = {
            '0': 0,
            '1-50': 0,
            '51-100': 0,
            '101-200': 0,
            '201-500': 0,
            '500+': 0,
        }

        for length in lengths:
            if length == 0:
                length_ranges['0'] += 1
            elif length <= 50:
                length_ranges['1-50'] += 1
            elif length <= 100:
                length_ranges['51-100'] += 1
            elif length <= 200:
                length_ranges['101-200'] += 1
            elif length <= 500:
                length_ranges['201-500'] += 1
            else:
                length_ranges['500+'] += 1

        for range_name, count in length_ranges.items():
            print(f"  {range_name:8s} 字符: {count:3d} 个案例")

    # 显示典型成功案例
    print(f"\n【典型成功案例】（按补全长度排序）")
    success_cases = []
    for task_key, result in data['results'].items():
        if result['success']:
            success_cases.append({
                'task': task_key,
                'length': result.get('completion_length', 0),
            })

    success_cases.sort(key=lambda x: x['length'])

    for case in success_cases[:10]:
        print(f"  • {case['task']:20s} - 补全长度: {case['length']} 字符")

    # 显示典型失败案例
    print(f"\n【典型失败案例】")
    failed_cases = []
    for task_key, result in data['results'].items():
        if not result['success']:
            failed_cases.append({
                'task': task_key,
                'error': result['error'][:80],
                'type': 'SYNTAX' if result['has_syntax_error'] else
                       'RUNTIME' if result['has_runtime_error'] else
                       'TEST FAIL',
            })

    for case in failed_cases[:20]:
        print(f"  • {case['task']:20s} - [{case['type']}] {case['error']}")

    print("\n" + "=" * 100)


# ============================================================================
# Part 4: 对比 Agentic 和 Assisted 的结果
# ============================================================================

def cmd_compare(args):
    """对比 Agentic 和 Assisted 两种模式的结果"""
    agentic_file = OUTPUT_BASE / "evaluation_results.json"
    assisted_file = OUTPUT_BASE / "assisted_evaluation_results.json"

    if not agentic_file.exists():
        print(f"警告: Agentic 评估结果不存在 ({agentic_file})")
        print("请先运行 'python agentic_eval.py eval'")
        return

    if not assisted_file.exists():
        print(f"警告: Assisted 评估结果不存在 ({assisted_file})")
        print("请先运行 'python assisted_eval.py eval'")
        return

    with open(agentic_file, 'r') as f:
        agentic_data = json.load(f)

    with open(assisted_file, 'r') as f:
        assisted_data = json.load(f)

    print("=" * 100)
    print("Agentic vs Assisted Coding - 对比分析")
    print("=" * 100)

    print(f"\n{'指标':<30s} {'Agentic':>15s} {'Assisted':>15s} {'差异':>15s}")
    print("-" * 80)

    # 基本统计对比
    agentic_success = agentic_data['success_count']
    assisted_success = assisted_data['success_count']
    agentic_rate = agentic_data['success_rate']
    assisted_rate = assisted_data['success_rate']

    print(f"{'总任务数':<30s} {agentic_data['total']:>15d} {assisted_data['total']:>15d}")
    print(f"{'成功任务数':<30s} {agentic_success:>15d} {assisted_success:>15d} {assisted_success - agentic_success:>+15d}")
    print(f"{'成功率 (%)':<30s} {agentic_rate:>15.2f} {assisted_rate:>15.2f} {assisted_rate - agentic_rate:>+15.2f}")

    # 找出 Agentic 成功但 Assisted 失败的案例
    print("\n" + "=" * 100)
    print("Agentic 成功但 Assisted 失败的案例")
    print("=" * 100)

    agentic_success_tasks = set()
    for task_key, result in agentic_data['results'].items():
        if result['success']:
            agentic_success_tasks.add(task_key)

    assisted_success_tasks = set()
    for task_key, result in assisted_data['results'].items():
        if result['success']:
            assisted_success_tasks.add(task_key)

    agentic_only = agentic_success_tasks - assisted_success_tasks
    assisted_only = assisted_success_tasks - agentic_success_tasks
    both_success = agentic_success_tasks & assisted_success_tasks
    both_failed = set(agentic_data['results'].keys()) - agentic_success_tasks - (assisted_success_tasks - agentic_success_tasks)

    print(f"\nAgentic 独有成功: {len(agentic_only)} 个")
    for task in sorted(agentic_only)[:20]:
        error = assisted_data['results'][task]['error'][:80]
        print(f"  • {task:20s} - {error}")

    print(f"\nAssisted 独有成功: {len(assisted_only)} 个")
    for task in sorted(assisted_only)[:20]:
        print(f"  • {task}")

    print(f"\n两者都成功: {len(both_success)} 个")
    print(f"两者都失败: {len(both_failed)} 个")

    # 保存对比结果
    comparison = {
        'agentic_success': agentic_success,
        'assisted_success': assisted_success,
        'agentic_rate': agentic_rate,
        'assisted_rate': assisted_rate,
        'agentic_only_success': list(agentic_only),
        'assisted_only_success': list(assisted_only),
        'both_success': list(both_success),
        'both_failed': list(both_failed),
    }

    output_file = OUTPUT_BASE / "agentic_vs_assisted.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    print(f"\n对比结果已保存到: {output_file}")

    print("\n" + "=" * 100)


# ============================================================================
# Main: 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Assisted Coding Evaluation Toolkit for HumanEval',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python assisted_eval.py eval              # 运行基础评估
  python assisted_eval.py analyze           # 分析失败案例
  python assisted_eval.py summary           # 生成统计报告
  python assisted_eval.py compare           # 对比 Agentic 和 Assisted
  python assisted_eval.py all               # 运行完整分析流程
        """
    )

    parser.add_argument('command', nargs='?',
                       choices=['eval', 'analyze', 'summary', 'compare', 'all'],
                       help='要执行的命令',
                       default='all')

    args = parser.parse_args()

    if args.command == 'eval':
        cmd_eval(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'summary':
        cmd_summary(args)
    elif args.command == 'compare':
        cmd_compare(args)
    elif args.command == 'all':
        print("=" * 80)
        print("运行完整 Assisted Coding 分析流程")
        print("=" * 80)
        print()
        cmd_eval(args)
        print("\n" + "=" * 80 + "\n")
        cmd_analyze(args)
        print("\n" + "=" * 80 + "\n")
        cmd_summary(args)
        print("\n" + "=" * 80 + "\n")
        cmd_compare(args)
        print("\n完整分析流程完成！")


if __name__ == '__main__':
    main()
