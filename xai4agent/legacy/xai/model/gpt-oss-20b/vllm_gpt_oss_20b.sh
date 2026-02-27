#!/bin/bash

# 启动命令，记得提前激活运行环境
# nohup bash /root/autodl-tmp/vllm/vllm_gpt_oss_20b.sh \
#   > /root/autodl-tmp/vllm/vllm_gpt_oss_20b.log 2>&1 &

# GPT-OSS 使用 Harmony 格式；vLLM 需使用 openai 工具解析器
# 官方建议：--tool-call-parser openai + --enable-auto-tool-choice
# 注意：函数调用只支持 tool_choice="auto"

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/gpt-oss-20b \
    --host 127.0.0.1 \
    --port 8001 \
    --served-model-name gpt-oss-20b \
    --trust-remote-code \
    --max-model-len 65536 \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser openai
