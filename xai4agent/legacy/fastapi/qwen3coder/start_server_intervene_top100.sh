#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/FastAPI/qwen3coder
MASK_FILE=$BASE/mask_top100_from_01-21-09.47.txt
LOG=$BASE/server_intervene_top100.log
PIDFILE=$BASE/server_intervene_top100.pid
MASK="$(cat "$MASK_FILE")"
echo "[$(date '+%F %T')] start server_intervene top100" >> "$LOG"
exec /root/miniconda3/envs/qwen/bin/python /root/autodl-tmp/FastAPI/qwen3coder/server_intervene.py "$MASK" --mask-mode keep_top8 --host 127.0.0.1 --port 8000 >> "$LOG" 2>&1
