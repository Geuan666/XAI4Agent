# XAI Hidden States Pipeline（Agentic vs Assisted）

这份 README 说明 `/root/autodl-tmp/xai/output/humaneval/hidden_states` 下的 `.pt` 文件是如何生成的，
并简述三个核心脚本的功能、输入输出与伪代码逻辑。

## 现有数据检查

- `/root/autodl-tmp/xai/output/humaneval/hidden_states` 下已有 164 个 `humaneval_*.pt` 文件（0~163）。
- 目录内还有 `figs_all/`, `figs_single/` 及 `figs_all.zip`（用于图像/分析缓存）。

## 产物目录

### 1) Hidden states
- 路径：`/root/autodl-tmp/xai/output/humaneval/hidden_states/*.pt`
- 每个文件对应一个 task（例如 `humaneval_0.pt`）。

#### `.pt` 文件内容结构（torch.save 的 dict）
```
{
  "key": "humaneval_0",
  "agentic": {
    "hidden_states": Tensor  # shape 见下
    # 可选：input_ids, attention_mask（需要 --store-ids）
  },
  "assisted": {
    "hidden_states": Tensor
    # 可选：input_ids, attention_mask（需要 --store-ids）
  },
  "token_meta": {
    "pad_token_id": int,
    "agentic_length": int,
    "assisted_length": int,
    "canonical_positions": [int, ...],          # canonical 非空白 token 的位置（agentic 坐标）
    "canonical_non_ws_tokens": [str, ...],
    "save_mode": "full|canonical|pooled",
    "segment_positions": {...},                 # user/docstring/solution 的分段位置
    "tool_positions": {...},                    # tool_call/tool_response 等位置
    "message_positions": {...},                 # system/user/assistant 边界
    "canonical_meta": {...}                     # fence_start/code_start 等
  }
}
```

#### hidden_states 的形状
- `save_mode=full`：`[num_layers, seq_len, hidden_size]`
- `save_mode=canonical`：`[num_layers, num_canonical_tokens, hidden_size]`
- `save_mode=pooled`：`[num_layers, hidden_size]`（对 canonical tokens 做均值）

> 说明：当前默认以 `float16` 写回（节省存储）。

---

## 生成流程总览

```
pair_build.py   -> pair_prompts.json
token_build.py  -> pair_tokens.json
pair_forward.py -> /root/autodl-tmp/xai/output/humaneval/hidden_states/*.pt
```

> 注意：脚本默认路径和你实际文件位置可能不同。
> 当前实际文件在：
> - `/root/autodl-tmp/xai/exp/pair/pair_prompts.json`
> - `/root/autodl-tmp/xai/exp/pair/pair_tokens.json`
> - `/root/autodl-tmp/xai/exp/pair/pair_forward.py`
> 如果用默认参数，请确认路径是否一致，或通过 `--pairs/--tokens` 显式指定。

---

## 脚本说明与伪代码

### A) `/root/autodl-tmp/xai/exp/pair/pair_build.py`
**用途**：构建“对齐的 agentic/assisted prompts”，并拼接 canonical solution。

**输入**：
- HumanEval parquet（含 canonical_solution）
- assisted 输出 JSON
- agentic 输出 JSON（含 message_trace）
- `server1.py`（用于真实 prompt 格式）
- `/root/autodl-tmp/xai/exp/agentic/agent.py`（tool schema）

**输出**：
- `pair_prompts.json`（每个样本一对 prompt）

**核心逻辑**：
- assisted：仅 user prompt -> build_prompt
- agentic：复现 message_trace + tool schema -> build_prompt
- 在 prompt 里插入 `#Function definition and docstring:` 标记
- 将 canonical solution 作为代码块接到末尾

**伪代码**：
```
for each task:
  assisted_prompt = build_assisted_prompt(user_prompt)
  agentic_prompt  = build_agentic_prompt(message_trace, tools)
  assisted_prompt = insert_marker(assisted_prompt)
  agentic_prompt  = insert_marker(agentic_prompt)
  continuation    = canonical_solution_as_code_block()
  write pair_prompts[task] = {
      assisted: assisted_prompt + continuation,
      agentic:  agentic_prompt  + continuation
  }
```

---

### B) `/root/autodl-tmp/xai/exp/pair/token_build.py`
**用途**：对 prompt 做 tokenize，并把 assisted prompt 对齐到 agentic 坐标系。

**输入**：
- `pair_prompts.json`

**输出**：
- `pair_tokens.json`

**核心逻辑**：
1) 用 fast tokenizer 获取 offset_mapping
2) 定位 assisted 的三个 segment（user/docstring/solution）
3) 在 agentic 中定位对应 segment 的 token 范围
4) 构造 `assisted_aligned_ids`（填充到 agentic 长度），并生成 mask
5) 计算 canonical 非空白 tokens 的位置（用于后续分析）

**伪代码**：
```
for each task in pair_prompts:
  agentic_ids, agentic_offsets = tokenize(agentic)
  assisted_ids, _              = tokenize(assisted)
  segments = split_assisted(assisted)

  segment_positions = find_segments_in_agentic(segments, agentic_offsets)
  aligned_ids, aligned_mask = place_segments_into_agentic_length()

  canonical_positions = find_non_ws_tokens_in_canonical_solution(agentic)

  write pair_tokens[task] = {
    agentic_ids,
    assisted_ids,
    assisted_aligned_ids,
    assisted_aligned_attention_mask,
    canonical_non_ws_positions,
    segment_positions,
    tool_positions,
    message_positions,
    ...
  }
```

---

### C) `/root/autodl-tmp/xai/exp/pair/pair_forward.py`
**用途**：使用 token_build 产生的对齐输入做 forward，保存 hidden states。

**输入**：
- `pair_tokens.json`（可缓存为 `pair_tokens.pt`）

**输出**：
- `/root/autodl-tmp/xai/output/humaneval/hidden_states/*.pt`
- 日志：`/root/autodl-tmp/xai/exp/pair/pair_forward.tsv`

**关键点**：
- agentic attention_mask：全 1
- assisted attention_mask：来自 `assisted_aligned_attention_mask`
- `output_hidden_states=True`
- 仅保存 hidden_states（不保存 logits/attention）

**伪代码**：
```
for each task in pair_tokens:
  if aligned_ok == false: skip/error
  agentic_hidden  = forward(agentic_ids, agentic_mask)
  assisted_hidden = forward(assisted_aligned_ids, assisted_mask)

  save {
    agentic.hidden_states,
    assisted.hidden_states,
    token_meta
  } -> hidden_states/humaneval_X.pt
```

---

## 常见注意点

1) assisted 对齐依赖 marker 与 offsets，如果 agentic prompt 构建变化，需重新跑 `pair_build.py`/`token_build.py`。
2) `canonical_non_ws_positions` 是后续差分分析的关键位置，请确保 mask=1。
3) `.pt` 文件较大（每个 250~450MB），保存 `full` 会更大。
4) `--store-ids` 会显著增加文件体积。

---

## 建议的重跑命令（按需）

```
/root/miniconda3/envs/qwen/bin/python /root/autodl-tmp/xai/exp/pair/pair_build.py
/root/miniconda3/envs/qwen/bin/python /root/autodl-tmp/xai/exp/pair/token_build.py
/root/miniconda3/envs/qwen/bin/python /root/autodl-tmp/xai/exp/pair/pair_forward.py \
  --tokens /root/autodl-tmp/xai/exp/pair/pair_tokens.json
```
