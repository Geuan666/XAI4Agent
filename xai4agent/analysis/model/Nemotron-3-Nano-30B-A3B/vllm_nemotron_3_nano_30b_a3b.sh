#!/bin/bash

# 启动命令，记得提前激活运行环境
# nohup bash /root/autodl-tmp/vllm/vllm_nemotron_3_nano_30b_a3b.sh \
#   > /root/autodl-tmp/vllm/vllm_nemotron_3_nano_30b_a3b.log 2>&1 &

# Nemotron-3-Nano-30B-A3B 在 vLLM 中推荐使用 qwen3_coder 工具解析器
# 官方示例：--tool-call-parser qwen3_coder + --enable-auto-tool-choice

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/Nemotron-3-Nano-30B-A3B \
    --host 127.0.0.1 \
    --port 8002 \
    --served-model-name nemotron-3-nano-30b-a3b \
    --trust-remote-code \
    --max-model-len 65536 \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
