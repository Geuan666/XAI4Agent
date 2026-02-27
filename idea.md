# XAI4Agent Idea 

## 1. 项目目标
在 **Assisted Coding** 与 **Agentic Coding** 的可控对照下，定位并验证：模型内部哪些层、哪些组件、哪些路由/注意力行为与工具使用能力相关。

## 2. 当前进展（已完成）
- 已完成 pair 构建：`agentic vs assisted` 在同一任务上的对齐输入。（pair构造是基于Qwen3coder-30B-A3B的,目前Qwen3-8B还未实现）
- 已完成 token 对齐与 canonical token 标注。（是基于Qwen3coder-30B-A3B的,目前Qwen3-8B还未实现）
- 已完成 hidden states 导出、NLL/LogitLens/Routing 差分分析。（是基于Qwen3coder-30B-A3B的,目前Qwen3-8B还未实现）

## 4. 下一阶段核心科学问题
1. 工具信息（tool-aware prompt）相比无工具信息（tool-blind prompt），在模型内部到底激活了哪些组件？
2. 这些组件是否只“相关”，还是在因果上“必要/充分”？
3. Top-N router mask 的结果，与梯度归因得到的重要组件是否一致？

## 5. 新实验主线（基于 Attribution Patching）
参考 `references/Attribution Patching.md`，新增一条标准主线：

### Stage A: Controlled Pair
- 构造两套严格可比输入：
  - `tool-aware`（包含工具调用相关信息）
  - `tool-blind`（不包含工具调用信息）
- 保持任务、目标答案、解码/teacher-forcing设置一致。

### Stage B: Attribution Scoring
- 定义目标度量：对于两个prompts对应的输出x1和x2，对于token序列，使用loss(x1-x2)作为度量，或者使用logits.
- 在 corrupted/tool-blind 侧做 backward，计算每个候选激活的 attribution 分数：
  - `score = <grad_corrupted, (act_clean - act_corrupted)>`
- 候选激活至少覆盖：
  - layer residual stream
  - attention output
  - MoE router logits / gate output

### Stage C: Ranking + Stability
- 对 layer/component 排名，取 Top-K。
- 跨样本 bootstrap，给每层/组件输出置信区间与稳定性排名。

### Stage D: Causal Validation
- 对 Top-K 候选做 activation patching / selective masking。
- 观察性能变化、NLL变化、通过率变化，确认“相关”是否变成“因果”。

### Stage E: Mechanism Linking
- 将 attribution Top-K 与已有 `router top-N ablation` 的关键点做交集分析：
  - 若高重叠：支持“router是核心机制”
  - 若低重叠：说明还存在非-router关键路径（如attention通路）
