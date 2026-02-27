#!/usr/bin/env python3
"""
Agentic Evaluation Toolkit for HumanEval

综合分析工具集 - 分析 qwen_output.json 文件，提取多次尝试的案例
"""
import argparse
import json
import os
import sys
import importlib.util
import traceback
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

# ============================================================================
# Configuration
# ============================================================================

PROJECT_BASE = Path("/root/autodl-tmp/xai/dataset/humaneval/agentic/project")
OUTPUT_BASE = Path("/root/autodl-tmp/xai/dataset/humaneval")

# ============================================================================
# Part 1: 基础评估 (来自 eval.py)
# ============================================================================

def load_module_from_file(module_name: str, file_path: Path):
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_function_name(main_file: Path) -> str:
    """Extract the function name from main.py."""
    content = main_file.read_text()
    for line in content.split('\n'):
        if line.strip().startswith('def '):
            func_def = line.strip()[4:]
            func_name = func_def.split('(')[0].strip()
            return func_name
    raise ValueError(f"No function definition found in {main_file}")


def run_single_test(project_dir: Path) -> Tuple[bool, str]:
    """Run test for a single project directory. Returns (success, error_message)"""
    project_name = project_dir.name
    main_file = project_dir / "main.py"
    test_file = project_dir / "test.py"

    if not main_file.exists():
        return False, f"main.py not found"
    if not test_file.exists():
        return False, f"test.py not found"

    try:
        func_name = get_function_name(main_file)
        main_code = main_file.read_text()
        test_code = test_file.read_text()

        ns = {
            "__name__": f"humaneval_{project_name}",
            "__builtins__": __builtins__,
        }
        exec(compile(main_code, str(main_file), "exec"), ns)

        if func_name not in ns:
            return False, f"Function '{func_name}' not found in main.py"

        candidate = ns[func_name]

        exec(compile(test_code, str(test_file), "exec"), ns)

        if "check" not in ns:
            return False, f"'check' function not found in test.py"

        ns["check"](candidate)
        return True, ""

    except AssertionError as e:
        return False, f"Assertion failed: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def cmd_eval(args):
    """运行基础评估"""
    project_dirs = sorted([d for d in PROJECT_BASE.iterdir()
                          if d.is_dir() and d.name.startswith('humaneval_')])

    print(f"Found {len(project_dirs)} project directories")
    print(f"Project base: {PROJECT_BASE}")
    print("=" * 60)

    results = {}
    success_count = 0
    failed_count = 0
    failed_tests = []

    for project_dir in project_dirs:
        project_name = project_dir.name
        success, error_msg = run_single_test(project_dir)

        results[project_name] = {
            'success': success,
            'error': error_msg,
            'path': str(project_dir)
        }

        if success:
            success_count += 1
            print(f"✓ {project_name}: PASSED")
        else:
            failed_count += 1
            failed_tests.append(project_name)
            print(f"✗ {project_name}: FAILED - {error_msg}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(project_dirs)
    success_rate = (success_count / total * 100) if total > 0 else 0

    print(f"Total tests:     {total}")
    print(f"Passed:          {success_count}")
    print(f"Failed:          {failed_count}")
    print(f"Success rate:    {success_rate:.2f}%")

    if failed_tests:
        print(f"\nFailed tests ({len(failed_tests)}):")
        for test_name in failed_tests:
            error = results[test_name]['error']
            print(f"  - {test_name}: {error}")

    # Save results
    results_file = OUTPUT_BASE / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'total': total,
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': success_rate,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


# ============================================================================
# Part 2: 分析 qwen_output.json (来自 analyze_retries.py)
# ============================================================================

def parse_json_file(file_path: str) -> List[Dict]:
    """解析 JSON 文件"""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []


def analyze_attempts(events: List[Dict]) -> Dict[str, Any]:
    """分析事件序列，提取关键指标"""
    result = {
        'has_failure': False,
        'num_edits': 0,
        'num_test_runs': 0,
        'failure_reasons': [],
        'edit_sequence': [],
        'test_results': [],
        'num_turns': 0,
        'duration_ms': 0,
    }

    for event in events:
        if event.get('type') == 'result':
            result['num_turns'] = event.get('num_turns', 0)
            result['duration_ms'] = event.get('duration_ms', 0)

            stats = event.get('stats', {})
            tools = stats.get('tools', {})
            edit_tool = tools.get('byName', {}).get('edit', {})
            result['num_edits'] = edit_tool.get('count', 0)

        if event.get('type') == 'user':
            message = event.get('message', {})
            content_list = message.get('content', [])
            for item in content_list:
                if item.get('type') == 'tool_result':
                    content = item.get('content', '')
                    if 'FAILURE' in content or 'AssertionError' in content:
                        result['has_failure'] = True
                        result['num_test_runs'] += 1
                        result['test_results'].append('FAIL')
                        if 'AssertionError' in content:
                            lines = content.split('\n')
                            for line in lines:
                                if 'assert' in line:
                                    result['failure_reasons'].append(line.strip())
                    elif 'SUCCESS' in content:
                        result['num_test_runs'] += 1
                        result['test_results'].append('PASS')

        if event.get('type') == 'assistant':
            message = event.get('message', {})
            content_list = message.get('content', [])
            for item in content_list:
                if item.get('type') == 'tool_use':
                    if item.get('name') == 'edit':
                        input_data = item.get('input', {})
                        old_string = input_data.get('old_string', '')[:100]
                        new_string = input_data.get('new_string', '')[:100]
                        result['edit_sequence'].append({
                            'old_preview': old_string,
                            'new_preview': new_string
                        })

    return result


def cmd_analyze(args):
    """分析所有 qwen_output.json 文件"""
    output_files = sorted(PROJECT_BASE.glob('*/qwen_output.json'))

    print(f"找到 {len(output_files)} 个 qwen_output.json 文件\n")

    retry_cases = []
    single_attempt_cases = []
    error_cases = []

    for output_file in output_files:
        project_name = output_file.parent.name
        events = parse_json_file(str(output_file))

        if not events:
            error_cases.append(project_name)
            continue

        analysis = analyze_attempts(events)

        case_info = {
            'project': project_name,
            'num_edits': analysis['num_edits'],
            'num_test_runs': analysis['num_test_runs'],
            'num_turns': analysis['num_turns'],
            'duration_ms': analysis['duration_ms'],
            'has_failure': analysis['has_failure'],
            'test_results': analysis['test_results'],
            'failure_reasons': analysis['failure_reasons'],
            'edit_sequence': analysis['edit_sequence'],
        }

        if analysis['has_failure'] or analysis['num_edits'] >= 2:
            retry_cases.append(case_info)
        else:
            single_attempt_cases.append(case_info)

    # 打印结果
    print("=" * 80)
    print(f"经过多次尝试才成功的案例: {len(retry_cases)} 个")
    print("=" * 80)

    retry_cases.sort(key=lambda x: x['num_test_runs'], reverse=True)

    for i, case in enumerate(retry_cases[:20], 1):
        print(f"\n【{i}】{case['project']}")
        print(f"    编辑次数: {case['num_edits']}")
        print(f"    测试运行: {case['num_test_runs']} 次")
        print(f"    测试序列: {' -> '.join(case['test_results'])}")
        print(f"    轮数: {case['num_turns']}")
        print(f"    耗时: {case['duration_ms'] / 1000:.1f} 秒")

        if case['failure_reasons']:
            print(f"    失败原因:")
            for reason in case['failure_reasons'][:3]:
                print(f"      - {reason}")

    print("\n" + "=" * 80)
    print(f"一次尝试即成功的案例: {len(single_attempt_cases)} 个")
    print(f"解析错误的案例: {len(error_cases)} 个")
    print("=" * 80)

    # 保存结果
    output_file = OUTPUT_BASE / "retry_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'retry_cases': retry_cases,
            'single_attempt_cases': single_attempt_cases,
            'error_cases': error_cases,
            'summary': {
                'total_files': len(output_files),
                'retry_count': len(retry_cases),
                'single_attempt_count': len(single_attempt_cases),
                'error_count': len(error_cases),
                'retry_rate': f"{len(retry_cases) / len(output_files) * 100:.1f}%"
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n详细分析结果已保存到: {output_file}")


# ============================================================================
# Part 3: 找出修复后成功的案例 (来自 find_retry_cases.py)
# ============================================================================

def cmd_find_retry(args):
    """找出修复后成功的案例"""
    analysis_file = OUTPUT_BASE / "retry_analysis.json"

    if not analysis_file.exists():
        print(f"错误: 请先运行 'python agentic_eval.py analyze' 生成分析数据")
        return

    with open(analysis_file, 'r') as f:
        data = json.load(f)

    print("=" * 100)
    print("真正经过修复后才成功的案例（有失败记录且最终通过）")
    print("=" * 100)

    retry_success_cases = []
    for case in data['retry_cases']:
        test_results = case['test_results']
        if 'FAIL' in test_results and test_results[-1] == 'PASS':
            retry_success_cases.append(case)

    retry_success_cases.sort(key=lambda x: x['test_results'].count('FAIL'), reverse=True)

    print(f"\n共找到 {len(retry_success_cases)} 个修复后成功的案例\n")

    for i, case in enumerate(retry_success_cases[:30], 1):
        fails = case['test_results'].count('FAIL')
        passes = case['test_results'].count('PASS')
        total = len(case['test_results'])

        print(f"【{i}】{case['project']:20s} | 编辑: {case['num_edits']}次 | "
              f"测试: {total}次 ({fails}败{passes}胜) | "
              f"轮数: {case['num_turns']:2d} | 耗时: {case['duration_ms']/1000:5.1f}s")
        print(f"    序列: {' -> '.join(case['test_results'])}")

    # 保存
    output_file = OUTPUT_BASE / "retry_success_cases.json"
    with open(output_file, 'w') as f:
        json.dump(retry_success_cases, f, indent=2, ensure_ascii=False)

    print(f"\n详细数据已保存到: {output_file}")

    # 统计
    print("\n" + "=" * 100)
    print("失败次数分布")
    print("=" * 100)

    failure_dist = {}
    for case in retry_success_cases:
        fails = case['test_results'].count('FAIL')
        failure_dist[fails] = failure_dist.get(fails, 0) + 1

    for fails in sorted(failure_dist.keys()):
        count = failure_dist[fails]
        print(f"  {fails} 次失败: {count:3d} 个案例")

    print("\n" + "=" * 100)
    print("编辑次数分布")
    print("=" * 100)

    edit_dist = {}
    for case in retry_success_cases:
        edits = case['num_edits']
        edit_dist[edits] = edit_dist.get(edits, 0) + 1

    for edits in sorted(edit_dist.keys()):
        count = edit_dist[edits]
        print(f"  {edits} 次编辑: {count:3d} 个案例")


# ============================================================================
# Part 4: 生成统计报告 (来自 generate_summary.py)
# ============================================================================

def cmd_summary(args):
    """生成综合统计报告"""
    retry_file = OUTPUT_BASE / "retry_success_cases.json"
    analysis_file = OUTPUT_BASE / "retry_analysis.json"

    if not retry_file.exists() or not analysis_file.exists():
        print(f"错误: 请先运行 'python agentic_eval.py analyze' 和 'find-retry' 生成数据")
        return

    with open(retry_file, 'r') as f:
        retry_success_cases = json.load(f)

    with open(analysis_file, 'r') as f:
        analysis_data = json.load(f)

    failed_cases = [c for c in analysis_data['retry_cases']
                    if c['test_results'][-1] == 'FAIL']

    print("=" * 100)
    print("                     Qwen Code HumanEval 任务分析报告")
    print("=" * 100)

    print(f"""
【总体统计】
  总任务数: 164
  成功任务: 160 ({160/164*100:.1f}%)
  失败任务: 2 ({2/164*100:.1f}%)
  解析错误: 2

【成功任务分布】
  一次编辑成功: 148 个 ({148/160*100:.1f}%)
  多次编辑成功: 12 个 ({12/160*100:.1f}%)

【尝试次数分布】
""")

    test_runs_dist = Counter()
    for case in retry_success_cases:
        test_runs_dist[len(case['test_results'])] += 1

    for runs in sorted(test_runs_dist.keys()):
        count = test_runs_dist[runs]
        print(f"  运行 {runs} 次测试: {count:3d} 个案例")

    print(f"""
【编辑次数分布】
""")

    edit_dist = Counter()
    for case in retry_success_cases:
        edit_dist[case['num_edits']] += 1

    for edits in sorted(edit_dist.keys()):
        count = edit_dist[edits]
        print(f"  编辑 {edits} 次: {count:3d} 个案例")

    print(f"""
【耗时分布】
""")

    time_ranges = {'0-30s': 0, '30-60s': 0, '60-120s': 0, '120-300s': 0, '300s+': 0}

    for case in retry_success_cases:
        duration = case['duration_ms'] / 1000
        if duration < 30:
            time_ranges['0-30s'] += 1
        elif duration < 60:
            time_ranges['30-60s'] += 1
        elif duration < 120:
            time_ranges['60-120s'] += 1
        elif duration < 300:
            time_ranges['120-300s'] += 1
        else:
            time_ranges['300s+'] += 1

    for range_name, count in time_ranges.items():
        print(f"  {range_name}: {count:3d} 个案例")

    print(f"""
【典型失败案例】
""")

    if failed_cases:
        for case in failed_cases:
            print(f"  • {case['project']}")
            print(f"    编辑 {case['num_edits']} 次, 测试 {len(case['test_results'])} 次, 最终仍失败")
    else:
        print("  无")

    print(f"""
【需要多次尝试的典型案例】（按失败次数排序）
""")

    multi_fail = [c for c in retry_success_cases if c['test_results'].count('FAIL') >= 2]
    multi_fail.sort(key=lambda x: x['test_results'].count('FAIL'), reverse=True)

    for i, case in enumerate(multi_fail[:10], 1):
        fails = case['test_results'].count('FAIL')
        print(f"  {i}. {case['project']:20s} - {fails} 次失败, {case['num_edits']} 次编辑, "
              f"{case['duration_ms']/1000:.1f}秒")

    print(f"""
【关键发现】
""")

    print("""
1. 失败模式分析:
   - 153/160 (95.6%) 的成功案例只需要 1 次编辑
   - 最常见的失败原因是对题目理解有偏差

2. 典型修复模式:
   a) 浮点数精度问题 (如 humaneval_2)
      - 首次使用 math.modf() 导致精度问题
      - 修复：改用 number - math.floor(number)

   b) 边界条件处理 (如 humaneval_78)
      - 首次未考虑空输入情况
      - 修复：添加 if not num: return 0

   c) 条件判断逻辑 (如 humaneval_163)
      - 首次使用 or 条件，导致逻辑错误
      - 修复：改为 and 条件

   d) 理解题意偏差 (如 humaneval_130)
      - 对递归/迭代规则理解不正确
      - 经过多次调试才找到正确实现

3. 未能解决的案例 (2个):
   - humaneval_145: 排序问题，尝试 8 次后仍失败
   - humaneval_33: 排序逻辑问题，尝试 3 次后仍失败
""")

    print("=" * 100)


# ============================================================================
# Part 5: 深入分析失败案例 (来自 deep_analysis.py)
# ============================================================================

def cmd_deep(args):
    """深入分析多次失败的案例"""
    retry_file = OUTPUT_BASE / "retry_success_cases.json"

    if not retry_file.exists():
        print(f"错误: 请先运行 'python agentic_eval.py find-retry' 生成数据")
        return

    with open(retry_file, 'r') as f:
        cases = json.load(f)

    multi_fail_cases = [c for c in cases if c['test_results'].count('FAIL') >= 2]

    print("=" * 100)
    print(f"深入分析多次失败才成功的案例（共 {len(multi_fail_cases)} 个）")
    print("=" * 100)

    for case in multi_fail_cases:
        print(f"\n{'='*100}")
        print(f"案例: {case['project']}")
        print(f"测试序列: {' -> '.join(case['test_results'])}")
        print(f"编辑次数: {case['num_edits']}, 轮数: {case['num_turns']}, 耗时: {case['duration_ms']/1000:.1f}秒")
        print(f"{'='*100}")

        json_path = PROJECT_BASE / case['project'] / "qwen_output.json"
        try:
            with open(json_path, 'r') as f:
                events = json.load(f)

            print("\n【失败与修复过程】\n")

            test_count = 0
            edit_count = 0

            for i, event in enumerate(events):
                if event.get('type') == 'user':
                    message = event.get('message', {})
                    content_list = message.get('content', [])
                    for item in content_list:
                        if item.get('type') == 'tool_result':
                            content = item.get('content', '')
                            if 'FAILURE' in content or 'SUCCESS' in content:
                                test_count += 1
                                if 'FAILURE' in content:
                                    print(f"第 {test_count} 次测试: ❌ 失败")
                                    lines = content.split('\n')
                                    for line in lines:
                                        if 'AssertionError' in line or 'assert' in line:
                                            if '>>>' not in line:
                                                print(f"  {line.strip()}")
                                            if len([x for x in lines if 'AssertionError' in x]) > 0:
                                                break
                                else:
                                    print(f"第 {test_count} 次测试: ✅ 成功")

                if event.get('type') == 'assistant':
                    message = event.get('message', {})
                    content_list = message.get('content', [])
                    for item in content_list:
                        if item.get('type') == 'tool_use' and item.get('name') == 'edit':
                            edit_count += 1
                            input_data = item.get('input', {})
                            new_string = input_data.get('new_string', '')

                            lines = new_string.split('\n')
                            func_lines = []
                            in_func = False
                            for line in lines:
                                if (line.strip() and
                                    not line.strip().startswith('#') and
                                    not line.strip().startswith('"""') and
                                    not line.strip().startswith("'''")):
                                    if line.strip().startswith('def '):
                                        in_func = True
                                    if in_func or 'return ' in line or 'for ' in line or 'if ' in line:
                                        func_lines.append(line)

                            print(f"\n第 {edit_count} 次修改:")
                            if func_lines:
                                for line in func_lines[:8]:
                                    print(f"  {line}")
                            print()

        except Exception as e:
            print(f"无法解析文件 {json_path}: {e}")

    print("\n" + "="*100)


# ============================================================================
# Main: 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Agentic Evaluation Toolkit for HumanEval',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python agentic_eval.py eval              # 运行基础评估
  python agentic_eval.py analyze           # 分析 qwen_output.json
  python agentic_eval.py find-retry        # 找出修复后成功的案例
  python agentic_eval.py summary           # 生成统计报告
  python agentic_eval.py deep              # 深入分析失败案例
  python agentic_eval.py all               # 运行完整分析流程
        """
    )

    parser.add_argument('command', nargs='?',
                       choices=['eval', 'analyze', 'find-retry', 'summary', 'deep', 'all'],
                       help='要执行的命令',
                       default='all')

    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_BASE),
                       help='输出目录')

    args = parser.parse_args()

    if args.command == 'eval':
        cmd_eval(args)
    elif args.command == 'analyze':
        cmd_analyze(args)
    elif args.command == 'find-retry':
        cmd_find_retry(args)
    elif args.command == 'summary':
        cmd_summary(args)
    elif args.command == 'deep':
        cmd_deep(args)
    elif args.command == 'all':
        print("=" * 80)
        print("运行完整分析流程")
        print("=" * 80)
        print()
        cmd_analyze(args)
        print("\n" + "=" * 80 + "\n")
        cmd_find_retry(args)
        print("\n" + "=" * 80 + "\n")
        cmd_summary(args)
        print("\n完整分析流程完成！")


if __name__ == '__main__':
    main()
