#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/xai/exp/real
START=$(awk -F'\t' 'NR>1{gsub("humaneval_","",$2); if(($2+0)>m)m=$2+0} END{print m+1}' real_log_mask_top100.tsv)
TS=$(date +%Y%m%d_%H%M%S)
cp real_output_mask_top100.json "real_output_mask_top100.backup_${TS}.json"
cp real_log_mask_top100.tsv "real_log_mask_top100.backup_${TS}.tsv"
echo "[START] $(date '+%F %T') start=${START}" | tee -a run_mask_top100_resume.out
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost OPENAI_BASE_URL=http://127.0.0.1:8000/v1 QWEN_MODEL=qwen3-coder-30b /root/miniconda3/envs/qwen/bin/python run.py \
  --project-root /root/autodl-tmp/xai/exp/real/projects \
  --start "${START}" \
  --output /root/autodl-tmp/xai/exp/real/real_output_mask_top100.json \
  --log /root/autodl-tmp/xai/exp/real/real_log_mask_top100_resume.tsv \
  2>&1 | tee -a run_mask_top100_resume.out
