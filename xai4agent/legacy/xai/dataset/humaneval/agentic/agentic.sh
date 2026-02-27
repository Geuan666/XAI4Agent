#!/usr/bin/env bash
set -euo pipefail

# ===== 配置 =====
BASE_DIR="/root/autodl-tmp/xai/dataset/humaneval/agentic/project"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/root/autodl-tmp/xai/dataset/humaneval/data"
LOG_FILE="${SCRIPT_DIR}/agentic_run_log.tsv"
WORKER_SCRIPT="${SCRIPT_DIR}/.agentic_worker.sh"

# 并行任务数（可修改为 4 或 6）
PARALLEL_JOBS=16

# 超时时间（秒）
TIMEOUT=600

# 模型配置
PROMPT='你正在处理 HumanEval 数据集的代码补全任务。

任务说明：
- 每个项目目录包含 main.py 和 test.py
- main.py 只有函数签名和文档字符串，函数体只有 pass
- 目标是补全函数体，使 python test.py 全部通过

流程要求：
1. run_shell_command 执行 "pwd && ls -la" 确认目录
2. run_shell_command 执行 "cat main.py" 读取函数签名和文档字符串
3. run_shell_command 执行 "cat test.py" 读取测试用例
4. todo_write 输出 3 到 5 条任务清单，包含需求要点、边界条件、实现步骤、验证步骤
5. 只根据 docstring 和 test.py 的要求实现最简正确逻辑，优先使用标准库
6. edit 把 main.py 中的 pass 替换为实现，保持函数签名、文档字符串、import 不变
7. run_shell_command 执行 "python test.py" 运行测试
8. 若测试失败，依据报错修改 main.py，再次测试，直到通过

注意事项：
- 只能修改 main.py，不允许修改 test.py 或新增文件
- 不执行 pip，不访问网络
- 不在回复中贴完整代码，所有改动用 edit 工具完成
- 通过测试后立即停止，不做额外改动,未通过测试则一直尝试修改代码直到通过测试'

# ===== 生成 Worker 脚本 =====
cat > "$WORKER_SCRIPT" << 'WORKER_EOF'
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$1"
PROJECT_NAME="$(basename "$PROJECT_DIR")"
LOG_FILE="$2"
PROMPT="$3"
TIMEOUT="$4"
JSON_OUTPUT="qwen_output.json"

# 进入项目目录
cd "$PROJECT_DIR" || {
  echo "$(date '+%F %T')|${PROJECT_NAME}|ERROR|Cannot enter directory|1" >> "$LOG_FILE"
  exit 1
}

# 检查必要文件
if [[ ! -f "main.py" ]]; then
  echo "$(date '+%F %T')|${PROJECT_NAME}|ERROR|main.py not found|1" >> "$LOG_FILE"
  exit 1
fi

# 记录开始时间
START_TIME="$(date '+%s')"
START_TIMESTAMP="$(date '+%F %T')"

# 运行模型补全代码
set +e
timeout "$TIMEOUT" qwen -p "$PROMPT" --yolo --output-format json > "$JSON_OUTPUT" 2>&1
EXIT_CODE=$?
set -e

# 只以测试结果为准，关闭 pass/docstring 预判
if python test.py >/dev/null 2>&1; then
  STATUS="SUCCESS"
  EXIT_CODE=0
else
  STATUS="FAIL_TEST"
  EXIT_CODE=1
fi

# 计算耗时
END_TIME="$(date '+%s')"
DURATION=$((END_TIME - START_TIME))

# 记录详细日志
echo "$(date '+%F %T')|${PROJECT_NAME}|${STATUS}|Exit:${EXIT_CODE}|Duration:${DURATION}s" >> "$LOG_FILE"

# 输出到控制台（带时间戳）
echo "[${START_TIMESTAMP}] ${PROJECT_NAME}: ${STATUS} (exit=${EXIT_CODE}, time=${DURATION}s)"

exit $EXIT_CODE
WORKER_EOF

chmod +x "$WORKER_SCRIPT"

# ===== 前置检查 =====
if [[ ! -d "$BASE_DIR" ]]; then
  echo "ERROR: BASE_DIR 不存在：$BASE_DIR" >&2
  exit 1
fi

# ===== 重新生成测试项目 =====
echo "============================================================"
echo "重新生成测试项目..."
echo "============================================================"
python3 "${DATA_DIR}/generate_projects.py"
echo ""

# ===== 准备任务列表 =====
shopt -s nullglob
dirs=("$BASE_DIR"/*/)

if [[ "${#dirs[@]}" -eq 0 ]]; then
  echo "ERROR: $BASE_DIR 下没有任何子文件夹可处理。" >&2
  exit 1
fi

# 创建临时文件存储任务列表
TASK_LIST="$(mktemp)"
for d in "${dirs[@]}"; do
  echo "${d%/}" >> "$TASK_LIST"
done

TOTAL_TASKS=$(wc -l < "$TASK_LIST")

# 初始化日志文件
echo -e "timestamp\tproject\tstatus\texit_code\tduration" > "$LOG_FILE"

echo "============================================================"
echo "HumanEval Agentic Code Completion - Parallel Execution"
echo "============================================================"
echo "Base Directory: $BASE_DIR"
echo "Parallel Jobs:   $PARALLEL_JOBS"
echo "Total Tasks:     $TOTAL_TASKS"
echo "Timeout:         ${TIMEOUT}s per task"
echo "Log File:        $LOG_FILE"
echo "============================================================"
echo "Started at:      $(date '+%F %T')"
echo ""

# 执行并行任务
START_TIME="$(date '+%s')"

export LOG_FILE PROMPT TIMEOUT

cat "$TASK_LIST" | xargs -P "$PARALLEL_JOBS" -I {} bash "$WORKER_SCRIPT" {} "$LOG_FILE" "$PROMPT" "$TIMEOUT"

# 等待所有任务完成
wait

END_TIME="$(date '+%s')"
TOTAL_DURATION=$((END_TIME - START_TIME))

# 统计结果
SUCCESS_COUNT=$(grep -c "|SUCCESS|" "$LOG_FILE" || echo "0")
FAIL_COUNT=$(grep -c "|FAIL|" "$LOG_FILE" || echo "0")
FAIL_NO_CODE_COUNT=$(grep -c "|FAIL_NO_CODE|" "$LOG_FILE" || echo "0")
FAIL_TEST_COUNT=$(grep -c "|FAIL_TEST|" "$LOG_FILE" || echo "0")
ERROR_COUNT=$(grep -c "|ERROR|" "$LOG_FILE" || echo "0")

# 汇总日志
echo ""
echo "============================================================"
echo "Execution Summary"
echo "============================================================"
echo "Total Tasks:      $TOTAL_TASKS"
echo "Successful:       $SUCCESS_COUNT"
echo "Failed:           $FAIL_COUNT"
echo "No Code Written:  $FAIL_NO_CODE_COUNT"
echo "Errors:           $ERROR_COUNT"
echo "Success Rate:     $(( SUCCESS_COUNT * 100 / TOTAL_TASKS ))%"
echo "Total Duration:   ${TOTAL_DURATION}s ($(( TOTAL_DURATION / 60 ))m $(( TOTAL_DURATION % 60 ))s)"
echo "Avg Time/Task:    $(( TOTAL_DURATION / TOTAL_TASKS ))s"
echo "Finished at:      $(date '+%F %T')"
echo "============================================================"
echo ""
echo "Log saved to: $LOG_FILE"
echo ""
echo "Recent failures (if any):"
grep -E "|FAIL|" "$LOG_FILE" | tail -5 || echo "None"

# 清理临时文件
rm -f "$TASK_LIST"
rm -f "$WORKER_SCRIPT"

exit 0
