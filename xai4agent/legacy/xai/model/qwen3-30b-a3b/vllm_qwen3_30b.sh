#!/bin/bash

# 启动命令，记得提前激活运行环境
# nohup bash /root/autodl-tmp/vllm/vllm_gpt_oss_20b.sh \
#   > /root/autodl-tmp/vllm/vllm_gpt_oss_20b.log 2>&1 &

# Qwen3 使用 Hermes 风格工具调用格式；vLLM 需使用 hermes 解析器。

python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models/qwen3-30b-a3b \
    --host 127.0.0.1 \
    --port 8003 \
    --served-model-name qwen3-30b-a3b \
    --trust-remote-code \
    --max-model-len 65536 \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
