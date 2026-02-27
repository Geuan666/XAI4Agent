# gpt-oss-20b prompt/token builder

## Files
- `vllm_gpt_oss_20b.sh`
  - vLLM 启动脚本（工具调用解析器：`openai`）
  - 默认端口：8001

- `run_agentic_trace.py`
  - 使用 vLLM OpenAI API 运行 agentic pipeline
  - 生成 `agentic_output.json`（包含 message_trace）
  - 输出日志 `agentic_log.tsv`

- `build_prompt_tokens.py`
  - 读取 `agentic_output.json` + `assisted_output.json`
  - 使用 `tokenizer.apply_chat_template` 生成 **agentic/assisted prompts**
  - 使用 OpenAI tools schema (`read_file`) 生成工具调用结构
  - 输出：
    - `pair_prompts.json`
    - `pair_tokens.json`

## Inputs
- Parquet: `/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet`
- Assisted outputs: `/root/autodl-tmp/xai/exp/assisted/assisted_output.json`

## Typical flow
1) 启动 vLLM
```bash
nohup bash /root/autodl-tmp/xai/model/gpt-oss-20b/vllm_gpt_oss_20b.sh \
  > /root/autodl-tmp/vllm/vllm_gpt_oss_20b.log 2>&1 &
```

2) 生成 message_trace
```bash
/root/miniconda3/envs/qwen/bin/python /root/autodl-tmp/xai/model/gpt-oss-20b/run_agentic_trace.py \
  --limit 1 --start 0 --max-steps 16
```

3) 生成 prompts/tokens
```bash
/root/miniconda3/envs/qwen/bin/python /root/autodl-tmp/xai/model/gpt-oss-20b/build_prompt_tokens.py
```

## Tool-call format
- 使用 **OpenAI tools schema**（Harmony 兼容）
- `tool_calls` 写入 assistant messages

## Change history
- 2026-01-19: 初版创建（添加 vLLM 启动脚本、agentic trace 生成、prompt/token 构建流程）
