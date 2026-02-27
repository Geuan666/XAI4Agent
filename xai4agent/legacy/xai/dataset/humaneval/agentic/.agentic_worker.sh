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
