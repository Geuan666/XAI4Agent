# MoE Routing 实验报告（最新）

> 更新时间：2026-01-18（基于本次运行输出）

本报告基于 **HumanEval** 数据集在 **Qwen3‑Coder‑30B‑A3B (MoE, 48 layers, 128 experts, top‑8)** 上的两条范式（**agentic vs assisted**）实验结果，结合两套分析脚本输出：

- `analyze_routing.py`（**routing 主分析**）
  - 图像：`/root/autodl-tmp/xai/output/humaneval/figures/routers/01-18-10.51`
  - 表格：`/root/autodl-tmp/xai/output/humaneval/tables/01-18-10.51`
- `analyze_routing_diag.py`（**routing 诊断**）
  - 图像：`/root/autodl-tmp/xai/output/humaneval/figures/routers_diag/01-18-11.10`
  - 表格：`/root/autodl-tmp/xai/output/humaneval/tables/01-18-11.10`

---

## 1. 实验流程与数据生成

### 1.1 prompt 对齐与 token 对齐流程

- **pair_build.py**（`/root/autodl-tmp/xai/exp/pair/pair_build.py`）
  - 根据 assisted/agentic 的真实交互轨迹构建统一 prompt。
  - assisted 仅保留 user prompt + canonical solution；
  - agentic 以工具调用 trace 作为上下文，并插入 `<tool_response>` 标记。

- **token_build.py**（`/root/autodl-tmp/xai/exp/pair/token_build.py`）
  - 使用 Qwen3‑Coder tokenizer 对 agentic/assisted prompt tokenize。
  - 通过 **segment 对齐** 将 assisted 内容对齐到 agentic token 座标系：
    - 输出 `assisted_aligned_ids` 和 `assisted_aligned_attention_mask`（mask=0 为 padding）
  - 生成 **canonical_non_ws_positions**（canonical solution 的非空白 token）。
  - 生成 **tool_positions**（schema / call / response 标记）。

- **pair_forward.py**（`/root/autodl-tmp/xai/exp/pair/pair_forward.py`）
  - 仅用于保存 hidden states（非本报告重点）。

---

## 2. Routing 分析脚本说明

### 2.1 analyze_routing.py（主分析）
- 读取 `pair_tokens.json` 中对齐后的 ids + mask。
- 对每层 router logits 计算：
  - TV distance（top8‑renorm）、Jaccard、intersection、top1/top5 match
  - usage 分布、entropy、top1/top5 share
  - tool_response docstring vs non‑doc 分离
  - tool‑expert leakage（Δu tool_non_doc vs Δu canonical）
- 输出 layer_metrics.csv、layer_joint_metrics.csv、top_experts_by_layer.csv、tool_expert_leakage.csv 等。

### 2.2 analyze_routing_diag.py（诊断）
- 增加：entropy、Neff、margin、top8 重归一化 JS 等诊断指标。
- 增加“距离 tool_response 的泄漏分析”。

---

## 3. 关键结果（来自真实输出）

> 结果来源：`/root/autodl-tmp/xai/output/humaneval/tables/01-18-10.51/layer_metrics.csv`、
> `tool_expert_leakage_summary.csv`、`/root/autodl-tmp/xai/output/humaneval/tables/01-18-11.10/leakage_vs_distance.csv`

### 3.1 Layer 级分歧集中在顶层
- **mean TV 最大层**：**46, 45, 44, 43, 47**
- **mean Jaccard 最低层**：**46, 45, 44, 43, 47**

说明：**agentic vs assisted 的 routing 分歧主要集中在模型顶层**。

### 3.2 tool_response 非 docstring 段差异显著更大
跨层平均 TV：
- tool_response overall：**0.094**
- tool_response **docstring**：**0.081**
- tool_response **non‑doc**：**0.156**

=> **non‑doc 段 routing 差异约为 docstring 的 2 倍**。

### 3.3 tool_schema / tool_call 分组无法比较（结构性原因）
统计中：
- count_tool_schema = 0
- count_tool_call = 0

原因是 assisted 侧没有 schema/call 这些 token（mask=0），
因此这两组无法进行对齐比较，**不是 bug**。

### 3.4 tool‑expert leakage（routing 泄漏）
`tool_expert_leakage_summary.csv` 显示：
- |r| 最大层：Layer 40 (r≈0.833), Layer 32 (r≈0.740), Layer 28 (r≈0.735)…

=> **工具非 doc 段的专家偏好与 canonical 路由差异存在显著相关性**，支持“tool routing → code routing 泄漏”。

### 3.5 距离 tool_response 的分歧趋势
`leakage_vs_distance.csv`（diag 输出）：
- 仅在距离区间 **[9,16] ~ [257,512]** 有有效 token
- TV 随距离略增（0.091 → 0.101），Jaccard 略降（0.848 → 0.833）

说明：**离 tool_response 越远，routing 差异略增**。

---

## 4. 关键图示（已插入）

### 4.1 Routing 分歧总览（TV 曲线）
![](/root/autodl-tmp/xai/output/humaneval/figures/routers/01-18-10.51/tv_curve.png)

### 4.2 tool_response docstring vs non‑doc 对比
![](/root/autodl-tmp/xai/output/humaneval/figures/routers/01-18-10.51/tv_tool_response_split.png)

### 4.3 tool vs canonical TV 对比
![](/root/autodl-tmp/xai/output/humaneval/figures/routers/01-18-10.51/tv_tool_vs_canonical.png)

### 4.4 tool‑expert leakage（示例层）
（Layer 40 示例，相关性最高）
![](/root/autodl-tmp/xai/output/humaneval/figures/routers/01-18-10.51/tool_expert_leakage_layer40.png)

### 4.5 距离 tool_response 的泄漏趋势（diag）
![](/root/autodl-tmp/xai/output/humaneval/figures/routers_diag/01-18-11.10/leakage_vs_distance.png)

### 4.6 诊断 TV / Jaccard 曲线（diag）
![](/root/autodl-tmp/xai/output/humaneval/figures/routers_diag/01-18-11.10/A1_tv_curves.png)
![](/root/autodl-tmp/xai/output/humaneval/figures/routers_diag/01-18-11.10/B1_jaccard.png)

---

## 5. 输出文件对照表（已修正路径）

### analyze_routing.py 输出
- 图像目录：`/root/autodl-tmp/xai/output/humaneval/figures/routers/01-18-10.51`
  - `tv_curve.png`, `tv_tool_vs_canonical.png`, `tv_tool_response_split.png`, `tool_expert_leakage_layerXX.png`, ...
- 表格目录：`/root/autodl-tmp/xai/output/humaneval/tables/01-18-10.51`
  - `layer_metrics.csv`, `layer_joint_metrics.csv`
  - `top_experts_by_layer.csv`（top‑64）
  - `tool_expert_leakage_by_layer.csv`, `tool_expert_leakage_summary.csv`

### analyze_routing_diag.py 输出
- 图像目录：`/root/autodl-tmp/xai/output/humaneval/figures/routers_diag/01-18-11.10`
  - `A1_tv_curves.png`, `B1_jaccard.png`, `leakage_vs_distance.png`, ...
- 表格目录：`/root/autodl-tmp/xai/output/humaneval/tables/01-18-11.10`
  - `top_experts_by_layer.csv`, `top_experts_global.csv`, `leakage_vs_distance.csv`

---

## 6. 结论小结（可用于汇报）

1. **routing 差异主要集中在顶层（43–47层）**，在多个指标中一致出现。
2. **tool_response 的非 docstring 段引入显著的 routing shift**。
3. **tool 专家偏好与 canonical routing 差异相关**，存在“tool→code 路由泄漏”的证据链。
4. **距离 tool_response 越远，routing 分歧略增**（诊断结果）。

---

如需添加新的图或调整解读，请告诉我具体想加入哪一类对比。  
本报告已按你指定的时间戳路径与真实输出文件修正。  

