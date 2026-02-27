# Analyze Pipeline (Core Version)

本目录目前推荐以两个核心脚本作为主入口：

1. `analyze_expert_core.py`
2. `analyze_nll_core.py`

目标是把之前分散在多个脚本里的关键分析流程统一成两条主线：

- **Expert / Routing 主线**：关注哪些 expert 被更多使用、agentic 与 assisted 的路由差异、可直接用于 mask 的 router 排序点集。
- **NLL / Logit-Lens 主线**：关注两种范式在各层的语言建模质量差异（NLL、logit-lens、可选 JS）。

---

## 1) `analyze_expert_core.py`

### 输入

- `--model_path`
  - MoE 模型目录（默认：`/root/autodl-tmp/models/qwen3-coder-30b`）
- `--data_path`
  - token 构造结果（默认：`/root/autodl-tmp/xai/exp/pair/pair_tokens.json`）
  - 每个 sample 至少应包含：
    - agentic token ids（如 `agentic_ids`）
    - assisted 对齐 token ids（如 `assisted_aligned_ids`）
    - `canonical_non_ws_positions`（或同义字段）
- `--out_dir`
  - 输出根目录（默认：`/root/autodl-tmp/xai/output/humaneval`）

### 主要输出

写到：`<out_dir>/figures/expert_core` 与 `<out_dir>/tables`

- 热力图：
  - `expert_usage_all_agentic.png`
  - `expert_usage_all_assisted.png`
  - `expert_usage_all_diff.png`
  - `expert_usage_last_agentic.png`
  - `expert_usage_last_assisted.png`
  - `expert_usage_last_diff.png`
  - `expert_usage_all_diff_thresholded.png`
  - `expert_usage_last_diff_thresholded.png`
- 路由曲线图：
  - `routing_tv_curve.png`
  - `routing_jaccard_curve.png`
  - `routing_intersection_curve.png`
- 表格与矩阵：
  - `expert_core_layer_metrics.csv`
  - `expert_core_matrices.npz`
  - `expert_core_summary.json`
- mask 点序列（按值降序）：
  - `last_token_agentic_minus_assisted_points.json`
  - `last_token_assisted_minus_agentic_points.json`
  - `last_token_agentic_points.json`

### 实现思路（高层）

- 对每个 sample 的 agentic / assisted 输入做前向，读取每层 router logits。
- 在 canonical_nonws 位置上提取 top-k expert，并进行归一化统计。
- 统计两种粒度：
  - all canonical_nonws tokens
  - last canonical_nonws token
- 在层维度上计算 routing 差异指标（TV / Jaccard / Intersection）。
- 对 `agentic - assisted` 矩阵做阈值化，直接输出离散差异图（集成了原 `plot_expert_usage_thresholded.py` 的核心功能）。
- 直接导出后续 mask sweep 可消费的 points.json。

### 常用命令

```bash
python /root/autodl-tmp/xai/exp/analyze/analyze_expert_core.py \
  --model_path /root/autodl-tmp/models/qwen3-coder-30b \
  --data_path /root/autodl-tmp/xai/exp/pair/pair_tokens.json \
  --out_dir /root/autodl-tmp/xai/output/humaneval \
  --device cuda \
  --top_k 8 \
  --threshold_percentile 99 \
  --threshold_min 0.005
```

---

## 2) `analyze_nll_core.py`

### 输入

- `--model_path`
  - 模型目录（用于提取 final norm + lm_head）
- `--pair_tokens_json`
  - 来自 token 对齐脚本的样本元数据（默认：`/root/autodl-tmp/xai/exp/pair/pair_tokens.json`）
- `--hidden_states_dir`
  - 每题 `.pt` hidden states（默认：`/root/autodl-tmp/xai/output/humaneval/hidden_states`）
  - 需要 full hidden states，且包含 agentic 与 assisted
- `--out_dir`
  - 输出根目录（默认：`/root/autodl-tmp/xai/output/humaneval`）

### 主要输出

写到：`<out_dir>/figures/nll_core` 与 `<out_dir>/tables`

- NLL 曲线：
  - `nll_all_agentic_vs_assisted.png`
  - `delta_nll_all.png`
  - `nll_last_agentic_vs_assisted.png`
  - `delta_nll_last.png`
- Logit-lens 曲线：
  - `logitlens_all_agentic_vs_assisted.png`
  - `delta_logitlens_all.png`
  - `logitlens_last_agentic_vs_assisted.png`
  - `delta_logitlens_last.png`
- 可选 JS 曲线（`--js_mode != none`）：
  - `js_all.png`
  - `js_last.png`
- 表格与矩阵：
  - `nll_core_layer_metrics.csv`
  - `nll_core_matrices.npz`
  - `nll_core_summary.json`

### 实现思路（高层）

- 从模型抽取“轻量读出头”：final norm + lm_head。
- 对 hidden states 执行分层读出，计算目标 token 的 log-prob：
  - `NLL = -log p(target)`
  - `logit-lens score = log p(target)`
- 分别在两种 token 视角聚合：
  - all canonical_nonws
  - last canonical_nonws
- 输出 agentic / assisted 两条曲线和差分曲线；可选计算 JS 分歧。

### 常用命令

```bash
python /root/autodl-tmp/xai/exp/analyze/analyze_nll_core.py \
  --model_path /root/autodl-tmp/models/qwen3-coder-30b \
  --pair_tokens_json /root/autodl-tmp/xai/exp/pair/pair_tokens.json \
  --hidden_states_dir /root/autodl-tmp/xai/output/humaneval/hidden_states \
  --out_dir /root/autodl-tmp/xai/output/humaneval \
  --device cuda \
  --dtype float16 \
  --chunk_size 256 \
  --js_mode none
```

---

## 目录调整记录

按你的要求，两个工具脚本已移出 `exp/analyze`：

- `generate_agentic_projects.py` -> `/root/autodl-tmp/xai/exp/agentic/generate_agentic_projects.py`
- `convert_code_contests_to_humaneval.py` -> `/root/autodl-tmp/xai/dataset/code_contests/code/convert_code_contests_to_humaneval.py`

---

## 推荐工作流

1. 先跑 `analyze_expert_core.py`，得到：
   - expert 使用热力图
   - 差分阈值图
   - 可直接 mask 的 points.json
2. 再跑 `analyze_nll_core.py`，验证：
   - 在 all / last token 视角下，分层 NLL 与 logit-lens 的变化
3. 用 points.json 对接 mask sweep，做性能退化曲线与机制解释闭环。

---

## 备注

- 若 `hidden_states_dir` 缺失 full hidden states，`analyze_nll_core.py` 无法产出有效结果。
- 若显存有限，可通过 `--max_samples`、`--chunk_size` 降低开销。
- 两个脚本默认输出到同一 `out_dir`，便于统一管理图表与表格。
