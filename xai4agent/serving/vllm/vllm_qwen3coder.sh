#!/bin/bash

#启动命令，记得提前激活qwen环境
#nohup bash /root/autodl-tmp/vllm/vllm_qwen3coder.sh \
# > /root/autodl-tmp/vllm/vllm_qwen3coder.log 2>&1 &

# 启动 vLLM 服务器
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
/root/miniconda3/envs/qwen/bin/python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/models \
    --host 127.0.0.1 \
    --port 8000 \
    --served-model-name qwen3-coder-30b Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --trust-remote-code \
    --max-model-len 163840 \
    --dtype auto \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
