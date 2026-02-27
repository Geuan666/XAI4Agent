# Prompt

你是一位大模型可解释性研究顾问，请基于以下项目背景与实验设置，输出一份**详尽、可操作但不包含代码**的可解释性实验方案。重点是“操作思路与流程设计”，而不是代码实现。

---

## 1. 项目背景与核心目标（非常详细）

我们在做一个对照实验，比较 **assisted coding** 与 **agentic coding** 这两种范式对大模型内部表示/行为的影响，并希望用可解释性分析方法给出“有说服力的差分图与结论”。

核心问题：

- 仅在“工具使用（tool use）”这一维度不同的情况下，模型内部表征和生成决策如何变化？
- 这些变化是否集中在特定层/阶段？
- 能否找到对 agentic 能力起关键作用的内部机制，从而提升模型的工具使用能力？

我们希望得到**能在论文中进一步深入探讨的差分图**（类似 logit lens 曲线、CKA 曲线、KL 曲线或热力图），并能够解释“工具使用导致的差异发生在何处”。

---

## 2. 模型与服务环境

- 基础模型：`qwen3-coder-30b`
- 推理服务：FastAPI + transformers（本地服务），已设置为**贪心解码**（`do_sample=False, temperature=0, top_p=1`）。
- 不使用系统提示词，所有指令都写在 user prompt 中。

---

## 3. 数据集

- 任务数据：HumanEval
- 数据文件：`/root/autodl-tmp/xai/dataset/humaneval/data/test-00000-of-00001.parquet`
- 每条样本包含：`prompt`（函数签名+docstring）、`canonical_solution`（标准函数体）、`test`（评测代码）。

---

## 4. Assisted pipeline（无工具）

- 入口脚本：`/root/autodl-tmp/xai/exp/assisted_run.py`
- prompt：仅包含指令 + 函数定义/文档字符串
- 模型输出：直接生成函数体文本
- 结果写入：`/root/autodl-tmp/xai/exp/assisted/assisted_output.json`
- 评测脚本：`/root/autodl-tmp/xai/exp/assisted_eval.py`

---

## 5. Agentic pipeline（有工具）

- 入口脚本：`/root/autodl-tmp/xai/exp/agentic_run.py`
- Agent 实现：`/root/autodl-tmp/xai/exp/agentic/agent.py`（LangGraph create_react_agent）
- 工具：**只允许 read_file**（禁止写文件与 shell）
- 任务流程：模型先调用 `read_file` 读取 `main.py`（包含函数签名+docstring），再输出函数体
- 结果写入：`/root/autodl-tmp/xai/exp/agentic/agentic_output.json`
- 评测脚本：`/root/autodl-tmp/xai/exp/agentic_eval.py`

Agentic 的 prompt 与 assisted 几乎一致，只增加了工具使用说明与项目路径要求。

---

## 6. 现有分析已完成的部分（你可以利用这些已跑出的中间结果）

- Logit lens（每层目标 token 概率）
- KL divergence（agentic vs assisted）
- CKA（基于最后一个 token 的 hidden states）
- Cosine similarity
- 这些结果已保存在：`/root/autodl-tmp/xai/output/humaneval/hidden_states/figs_all`、`/root/autodl-tmp/xai/output/humaneval/hidden_states`

**当前问题：**

- 目标 token 选取不够理想（常为缩进空白 token），导致 logit lens 曲线不够直观。
- Agentic prompt 更长，RoPE 位置差可能影响中后层对齐。
- CKA 曲线中后层有非单调下降，可能是位置差或工具吸收导致。

---

## 7. 你的任务（请输出详细、可执行的“实验方案”）

请提供一份**系统化的可解释性实验方案**，包含但不限于以下内容（不需要代码，重点是“怎么做”）：

### A. 需要比较的内部信号

- 哪些 hidden states/位置最合理？例如：
  - 只看“最后生成位置”
  - 看“canonical_solution 前 N 个非空 token 的平均”
  - 看“function body span 的 token 表示”
- 是否需要对齐 token 位置？如何对齐？

### B. 建议采用的可解释性方法（至少 3-4 种）

- CKA：应对齐哪些 token？如何做 bootstrap 或置信区间？
- KL：在哪些位置计算？如何避免最后一层尖峰主导？
- Logit lens：用什么目标 token 更合理？如何做多 token 平均？
- Attention/Attribution：是否值得看 tool-response token 的注意力占比？
- 其他可选：SVCCA、Representation Sim, Head-level analysis 等

### C. 可视化/对比图形式（能上组会的）

- 双曲线对比图（agentic vs assisted）
- 差值曲线（agentic-assist）
- Heatmap（样本×层）
- 带 CI 的曲线

### D. 是否需要 hook 工具（transformer-lens / pyvene）

- 什么时候**必须**用 hook？
- 什么时候仅用 `output_hidden_states` 足够？
- 如果用 pyvene，能得到哪些“因果性”结论？

### E. 相关工作检索建议（请列出工作或方法名称）

- 例如：Logit Lens、Activation Patching、Causal Tracing、EAP、ACDC、ROME/MEMIT、Attention Rollout 等
- 希望你说明这些工作在人工智能领域的顶会如ICML、ICLR中通常怎么使用这些可解释性的工具

---

## 8. 输出要求

- **只要方案与思路，不要代码**
- 需要完整、可操作、结构清晰
- 给出明确的实验步骤建议（例如：先做哪张图，再做哪张图）
- 语言使用中文
- 先给出5种详尽且可实现、有一定效果的方案

# response

下面先给出 **5 套“详尽、可操作、能产出差分图”的实验方案**（每套都覆盖：内部信号怎么取、怎么对齐、用哪些方法、先画什么图、最后能得出什么结论）。随后我再给一个 **总览版 A–E 清单**，把关键决策点统一整理，方便你们按图索骥推进到论文级证据链。

---

## 方案 1：Canonical-solution 对齐的「时间×层」差分（把 logit lens / KL / CKA 做到“可解释”）

### 核心目的

你们现在的 logit lens 之所以“不直观”，根因是 **目标 token 常落在缩进/空白**；再加上 agentic prompt 更长导致 RoPE 位置差，**只看“最后一个 token”** 的 CKA 也很容易被“对齐错位”污染。

这一套方案的目标是：
把比较对象从“最后 token”升级为 **canonical_solution 的语义 token 序列**，直接产出论文里最有说服力的图：

- **层×时间（token index）** 的 KL/Δlogprob 热力图
- **分层曲线 + 置信区间**（不再被最后层尖峰主导）
- 还能自然回答“差异发生在何处（哪些层 + 哪些生成阶段）”。

### 操作流程（无代码版）

#### Step 1：统一用 teacher-forcing 让两种范式“预测同一条答案”

对每个 HumanEval 样本，构造两种上下文：

- assisted 上下文：你们 assisted_run 的 prompt
- agentic 上下文：**最终生成函数体时模型看到的完整上下文**（包含 tool call/response 的那一步上下文；务必以实际 agent trace 为准，而不是“想当然的模板”）

然后做一件关键事：
把 `canonical_solution`（建议仅取函数体部分）按 tokenizer 切成 token 序列，**追加到上下文后面**，用 teacher forcing 跑前向：

- 在 canonical 的每个位置 t，上下文都相同（都是 canonical 前缀），因此两范式差异更聚焦于“内部表征与决策机制”，而不是“输出文本跑偏导致的比较失真”。

#### Step 2：定义“语义 token 集”，彻底规避缩进空白

把 canonical token 按字符串规则分组并过滤：

- **过滤集**：只包含空白/缩进/纯换行的 token（例如解码后 strip 为空）
- **语义集**（建议至少三类分别统计）：

  1. **关键词/结构**：`def / return / if / for / while / try / except / import / from / in / is / not` 等
  2. **标识符**：函数名、参数名、局部变量、属性名
  3. **字面量与运算符**：数字、字符串、`+ - * / % == <=` 等
     （你们会得到“差异到底是在结构规划还是在细节 token”这一层解释。）

#### Step 3：把 logit lens 从“单 token 概率”升级为“多 token 负对数似然”

对每层 l、每个 canonical 位置 t：

- 计算 **下一 token（canonical[t+1]）** 在该层 logit lens 下的 logprob
- 对语义 token 集做聚合：

  - **曲线 1**：每层平均 logprob（或 NLL）
  - **曲线 2**：每层在 3 类 token 上分别的平均 NLL
    这会比“目标 token 概率”稳定得多，也更适合跨样本平均。

> 你们现在图里 assisted 在最后层突然抬升、agentic 接近 0，本质就是“目标 token 不对”。改成“canonical 语义 token 的平均 NLL”，曲线会立刻变得可读。

#### Step 4：KL/JS 的计算位置与“避免尖峰主导”

你们现在 KL 最后一层巨大尖峰（很常见）：最后层 logits 更尖锐，任何小差异都会放大。解决方法（选一种主用 + 一种做鲁棒性对照）：

- **主用：JS divergence（对称、有上界）**，更适合可视化比较
- 或者：KL 但做 **top-k union 限制**（在 agentic/assisted 的 top-k + canonical token 的并集上重归一化再算 KL），避免尾部分布数值噪声主导

#### Step 5：画图顺序（建议你们按这个出组会图）

1. **热力图**：x=canonical token index（仅语义 token），y=layer，颜色=ΔNLL(agentic-assisted) 或 JS

   - 一眼看出“差异发生在生成早期/中期/后期”
2. **分层曲线**：每层平均 ΔNLL / JS，带 95% CI（bootstrap over tasks）
3. **分 token 类别曲线**：关键词 vs 标识符 vs 运算符/字面量
4. **样本×层热力图**：每个样本的层均值 ΔNLL（或 JS），看异质性与长尾

### 你会得到的“可写进论文的结论口径”

- “工具使用范式的影响主要集中在 **第 X–Y 层**，并且在 **生成的某个阶段（例如前 20 个语义 token）** 最明显。”
- “差异主要体现在 **结构 token（return/if）** 还是 **标识符 token**，从而推断 agentic 影响的是全局规划还是局部细节。”

---

## 方案 2：长度与 RoPE 位置差的「四象限控制实验」——把 confound 拆干净（差分中的差分）

### 核心目的

你们已经明确担心：agentic prompt 更长 → RoPE 位置差 → 中后层对齐污染。
这套方案的目标是：**把“工具使用”与“长度/格式/位置差”做因子拆解**，最后用 **Difference-in-Differences** 给出非常干净的因果式证据。

#### 关键设计：2×2 或 2×3 对照组

至少做 4 个条件（推荐 6 个更强）：

#### 必做 4 条

1. **A0：assisted-short**（原始 assisted prompt）
2. **A1：assisted-long scaffold**（把 agentic 的 ReAct/tool 协议文本、路径要求等“脚手架”加进来，但不实际调用工具；长度尽量匹配 agentic）
3. **B0：agentic-real**（真实 agentic trace：tool call + tool response + 最终回答）
4. **B1：agentic-matched-no-tool**（保留同等长度/格式，但把 tool response 内容替换为“中性占位”（长度相同、信息量低），或替换为“与 assisted 已给信息完全重复”的版本）

#### 可选增强 2 条（强烈建议）

1. **A2：assisted-dup-spec**（把函数签名+docstring 重复一次，模拟“读文件后又看到一遍 spec”的效果）
2. **B2：agentic-trimmed**（把 trace 中与最终回答无关的历史内容裁剪掉，使 spec 起始位置与 assisted 对齐，测试 RoPE 敏感性）

### 操作流程

1. 先用方案 1 的 **canonical 对齐 NLL/JS** 做所有条件的指标
2. 做 **差分中的差分**：

   - 工具纯效应（去掉长度/格式）：
     [
     \Delta_{\text{tool}} = (B0 - A1)
     ]
   - 长度/脚手架效应：
     [
     \Delta_{\text{scaffold}} = (A1 - A0)
     ]
   - 最终你们要在论文里强调的“工具机制效应”：
     [
     \Delta_{\text{net}} = \Delta_{\text{tool}} - \Delta_{\text{scaffold}}
     ]
3. 同样对 CKA/KL/注意力做这个 DoD，会非常有说服力。

### 推荐可视化（组会/论文都够用）

- 4 条曲线叠图：A0/A1/B0/B1（每层 NLL 或 JS + CI）
- **DoD 差值曲线**：Δ_net 随 layer 变化
- 关键层段做放大图（例如你们现在 KL 在后几层抬升、CKA 在 30+ 层掉下去的区域）

### 你会得到的“可写进论文的结论口径”

- “即使控制 prompt 长度与协议格式，agentic 仍在 **第 X–Y 层** 出现显著差异（Δ_net ≠ 0），说明差异不是 RoPE/长度伪影，而是工具范式引入的内部机制改变。”

---

## 方案 3：注意力与信息来源分解（tool-response 是否真的被“读了”？读发生在哪些层/哪些头）

### 核心目的

你们要回答的核心问题之一是：

> “仅在 tool use 不同的情况下，模型内部表征/决策如何变化？变化集中在何处？”

注意力分析能提供非常直观的证据：

- agentic 在生成函数体时，是否把注意力显著分配给 **tool response 段**？
- 哪些层/哪些头最依赖 tool response？是否存在“检索头/定位头”？

### 操作流程

#### Step 1：把输入分段（segment）

对 agentic-real 上下文，把 token index 切成段（至少）：

- S1：任务指令（instruction）
- S2：spec（函数签名+docstring）
- S3：tool call 协议段（Action/Observation 等）
- S4：tool response 段（read_file 返回的 main.py 内容）
- S5：生成输出段（function body，按生成时刻 t）

assisted 的上下文也做同样分段（没有 S3/S4 时置空），或者用方案 2 的 scaffold 版本让段结构一致。

#### Step 2：定义“注意力质量指标”（比画一张 attention map 更可比）

对每个 layer、head、每个输出位置 t（建议只取**语义 token**位置）：

- 计算该 head 从输出位置 t 指向各 segment 的注意力权重总和

  - 例如：AttentionMass_to_ToolResponse(l,h,t) = sum_{i in S4} attn(l,h,t,i)

聚合方式建议三种都做（信息量不同）：

1. **全局平均**：对 t 取平均 → 得到 layer×head 的一个数
2. **随时间曲线**：按输出 token index 分桶（前 10/中间/后 10）
3. **按 token 类别**：关键词 vs 标识符 vs 运算符 的 attention mass

#### Step 3：找“关键头”，做可解释案例

- 在 layer×head 上找 Δattention_mass 最大的头（agentic - assisted/scaffold）
- 对这些头挑 3–5 个代表样本画 **输入带分段标注的 attention 可视化**
  （论文里常见做法：展示“该头从 return/if 等 token 强烈指向 tool response 里的函数签名/参数名”）

### 推荐图（非常适合组会）

- **layer×head 热力图**：ΔAttentionMass_to_ToolResponse
- **layer 曲线**：每层所有 head 的 tool-response attention mass 总和（或 top-10 heads 的总和）
- **分时间曲线**：生成早期 vs 后期，tool attention 是否不同（很多“规划”发生在早期）

### 你会得到的结论口径

- “agentic 在 **第 X–Y 层** 出现一组头，显著把注意力指向 tool response（而 scaffold 控制组没有），说明模型在这些层将 tool 信息整合进生成决策。”
- 如果发现 attention 主要在早期集中：可以推断 agentic 影响的是“计划/约束注入”；若在中后期集中：更像“细节对齐/变量拷贝”。

---

## 方案 4：因果定位（Activation Patching / Causal Tracing / Head-level Path Patching）——把“相关”升级为“因果”

### 核心目的

CKA、KL、注意力都还是“相关性证据”。你们如果想写到“关键机制/关键层”，需要至少一条因果链：

- 把 agentic 的某层激活 patch 到 assisted（或反过来），能否**显著改变** canonical token 的概率 / 生成结果？

这一步是论文说服力跃迁点。

### 你们需要先定义一个“干净的因果指标”

强烈建议用 **可微且稳定** 的指标，而不是 pass/fail（pass/fail 太离散、噪声大）：

- **Metric M1：canonical 语义 token 的平均 NLL**（来自方案 1）
- **Metric M2：关键结构 token 的 logprob**（如 `return`、关键分支条件运算符）
- 若要直接对“工具调用决策”做因果：
  **Metric M3：在应当发起 tool call 的位置，对“Action: read_file”序列的平均 logprob**
  （即对 agentic 的“决定调用工具那一步”做 teacher-forcing 评估）

### Patch 的分层推进路线（建议按难度递增）

### Phase 1：Layer-level residual patch（最先做，最快定位层段）

- 在同一输入（最好用方#案 2 的 matched scaffold 对照）下：

  1. 跑 assisted 与 agentic，缓存每层 residual stream（或 block 输出）在关键位置的激活
  2. 把 agentic 的第 l 层激活 patch 到 assisted，对比 M1/M2 的变化
- 扫 layer → 得到 **layer 重要性曲线/热力图**

### Phase 2：Position×Layer patch（定位“发生在输入的哪一段”）

位置建议至少三类：

- spec 段（函数签名/docstring 的关键 token）
- tool response 段（若存在）
- 输出段早期（前 N 个语义 token，通常是规划最明显）

扫 position×layer → 得到 **二维因果热力图**（这张图非常“论文脸”）

### Phase 3：Head-level / MLP-level path patch（定位到组件）

一旦你知道关键层段（比如你们现在的图暗示 30+ 或 40+ 层可能更不同）：

- 分别 patch 注意力头输出、MLP 输出
- 找出 top-K 头/MLP，对 M1/M2 的影响最大者
- 把这些头与方案 3 的“tool-response attention 高的头”交叉验证：
  同一批头同时满足 **(a) 强因果效应** 与 **(b) 强 tool-attention**，基本就可以称为“工具整合电路候选”。

> 如果你们愿意更进一步：用 **EAP（Edge Attribution Patching）/ ACDC（自动电路发现）** 从“哪些边重要”角度压缩成一个小电路，在顶会风格里非常常见。

## 推荐图

- layer 重要性曲线（patch effect vs layer）
- layer×position patch 热力图（最强）
- top heads 条形图 + 若干案例 attention map（与方案 3 串起来）

## 你会得到的结论口径

- “将 agentic 的第 X–Y 层（在 spec/tool-response 位置）的激活注入 assisted，可显著降低 canonical NLL/提升关键 token 概率，说明该层段携带了工具范式引入的关键计算；进一步定位到若干注意力头/MLP，是工具整合的因果载体。”

---

# 方案 5：按行为结果分组 + 表示空间分析（把“差异”与“agentic 是否带来收益”绑定）

## 核心目的

HumanEval 164 个样本差异很大：有些任务 agentic 可能更好，有些更差。
如果你们只看全体均值，很容易把“有效机制”与“无关差异/噪声”混在一起。

这套方案的目标是：
**把内部差分与外部收益对齐**，回答：“哪些内部差异真正与 agentic 能力提升相关？”

## 操作流程

### Step 1：用评测结果把样本分 4 组

基于你们现成的 `assisted_eval.py / agentic_eval.py` 输出，把每题分成：

- G11：都通过
- G10：agentic 通过、assisted 不通过（agentic 有收益）
- G01：assisted 通过、agentic 不通过（agentic 有代价）
- G00：都不通过

### Step 2：对每组分别画方案 1/3 的关键曲线

至少画三类组内均值 + CI：

- 每层 ΔNLL / JS（canonical 语义 token）
- 每层 CKA（但要用“span pooling”的 CKA，而不是 last token，见后面总览）
- 每层 tool-response attention mass（若用 scaffold 对齐更好）

你会马上看到：

- “真正与收益相关”的差异，应该在 G10 明显，而在 G11/G00 不一定明显。

### Step 3：做表示空间相似度的“更硬核版本”（可选但很加分）

对选定 span（如输出前 64 个语义 token 的 hidden states，或 spec 段）做：

- **SVCCA / PWCCA / Procrustes**：比较 agentic vs assisted 在每层的子空间相似性
- 或做一个 **cross-layer similarity matrix**：看 agentic 的第 l 层是否更像 assisted 的第 l+Δ 层（揭示“计算时序被推迟/提前”）

### Step 4：线性 probe（可选，解释“模型是否学会区分工具信息源”）

构造简单标签（不需要额外数据）：

- token 属于哪个 segment（instruction/spec/tool response/output）
- token 是否为关键词/标识符/字面量
- token 是否处在函数体早期（规划区）/后期（细节区）

在每层 hidden states 上训练线性 probe（只做分析，不是训练模型），比较 agentic vs assisted 在这些标签上的可分性随层变化。
如果 agentic 的某些层显著更可分，且与 G10 强相关，就是很好的机制线索。

## 推荐图

- 4 组分层曲线对比（最直观）
- G10 vs G01 的差分中的差分（收益 vs 代价机制分离）
- cross-layer CKA 矩阵（可上论文）

## 你会得到的结论口径

- “与 agentic 带来收益（G10）强相关的内部差异集中在第 X–Y 层，并表现为更强的 tool 信息整合/更低的 canonical NLL；而导致代价（G01）的差异集中在另一层段或另一类 token/头。”

---

# 总览：把你提的 A–E 要求统一落到可执行决策点

下面是更“规范化”的清单，你们可以把它当作整个项目的实验 SOP。

---

## A. 需要比较的内部信号：取哪些 hidden states/位置最合理？如何对齐？

### A1. 强烈推荐的三个“位置定义”（按优先级）

1. **canonical_solution 的语义 token 序列（teacher forcing）**

   - 最适合 logit lens / KL / 分阶段差分
   - 解决“目标 token 是空白”的根问题
2. **function body span pooling（输出前 N 个语义 token 的均值/中位数）**

   - 最适合 CKA / SVCCA：把 per-token 噪声平均掉
   - 比 last token 稳定太多
3. **关键锚点 token（anchor points）**

   - 例如：第一个 `return`、第一个分支 `if`、第一个循环 `for`、第一个调用被测函数名/参数名
   - 适合做“机制解释案例”与 patching

### A2. token/位置对齐的三种层次（建议都做，形成证据链）

1. **语义对齐（canonical index 对齐）**：比较 canonical 第 t 个语义 token 在两范式下的内部状态
2. **段落对齐（segment 对齐）**：instruction/spec/tool response/output 四段分别统计
3. **位置对齐（RoPE 控制）**：用 scaffold/padding/trimming 让 spec 起始位置尽量对齐

   - 这一步是避免你们现在 CKA 中后层“非单调波动”被解读成 RoPE 伪影的关键

> 你们当前 CKA 在 30+ 层明显下滑、并伴随局部回升，很像“对齐问题 + 真差异混合”的信号；做完 scaffold/对齐后，这种曲线会更可解释。

---

## B. 建议采用的可解释性方法（至少 3–4 种，且每种怎么做更“论文级”）

### B1. CKA（你们已有，但建议升级）

- **不要只用最后 token**：改为对 function body 的前 N 个语义 token 做 pooling 后再算 CKA
- 额外做一个 **cross-layer CKA 矩阵**：看是否存在层对层“偏移匹配”（agentic 的层计算被推迟/提前）
- **置信区间**：对 164 题做 bootstrap（按题重采样），画 95% CI

### B2. KL / JS（你们已有，但建议换计算口径以避免最后层尖峰）

- 推荐主用 **JS divergence**（对称、有界、可视化更稳定）
- 或用 **top-k union KL**（减少尾部噪声）
- **计算位置**：在 canonical teacher forcing 的每个语义 token 上算，再对 token 平均/分段平均
  （不要只在“最后 token”算，否则会被生成偏差放大）

### B3. Logit lens（你们已有，但目标 token 必须重定义）

- 把“单目标 token 概率”改成：

  - canonical 语义 token 的 **平均 NLL / 平均 logprob**（更稳定）
  - 或者“关键 token 集”的平均 logprob（return/if/运算符）
- 如你们想更严谨：可以加一个 **Tuned Lens** 作为对照（logit lens 可能因中间层未对齐而失真）

### B4. Attention / Attribution（建议至少做注意力分段；梯度归因可选）

- **注意力分段占比**（方案 3）：layer×head 的 tool-response attention mass
- 进阶（可选）：

  - attention rollout / attention flow（给“信息从 tool 段流到输出 token”的整体证据）
  - 梯度类 attribution（Integrated Gradients / Grad×Act）用于补充“注意力不等于因果”的争议点

### B5. 其他可选（加分项）

- **SVCCA / PWCCA / Procrustes**：做表示子空间相似度（尤其配合分组 G10/G01 很有说服力）
- **Head-level analysis**：结合 patching，定位“工具整合头”
- **EAP / ACDC**：自动化缩小电路（如果你们目标是“找到关键机制并提升工具能力”，这是最贴合的路线）

---

## C. 可视化/对比图形式（能上组会、也能进论文）

按“信息密度/说服力”排序推荐你们优先产出：

1. **层×token-index 热力图**（canonical 语义 token）：ΔNLL 或 JS
2. **分层曲线 + 95% CI**：ΔNLL/JS/CKA（span pooling 后）
3. **差值曲线（agentic - assisted）+ DoD 曲线（去 confound）**
4. **样本×层热力图**：看异质性与长尾（并用 G10/G01 分组着色/分面）
5. **layer×head 热力图**：Δtool-attention mass（非常直观）
6. **causal patching 热力图**：layer×position 的因果效应（论文杀手锏）

---

## D. 是否需要 hook（transformer-lens / pyvene）？什么时候必须？

### D1. 仅用 `output_hidden_states` 就足够的场景

- span pooling 表示（用于 CKA/SVCCA/余弦）
- logit lens 的“中间层 unembed”（如果你们已有实现路径）
- teacher forcing 下的 per-token hidden/logits 统计

### D2. 必须用 hook（或类似机制）的场景

- **Activation Patching / Causal Tracing / Path Patching**（方案 4）：需要在 forward 中替换某层/某头的激活
- 想拿到 **head output / MLP output**（不仅是 attention 权重）来做组件级归因
- 想做更细粒度的“电路边”分析（EAP/ACDC）

### D3. 为什么建议用 pyvene（就算不用 transformer-lens）

- 你们是 HuggingFace/transformers 本地服务，pyvene 的干预式 API 更贴近这个栈
- 能直接支持“替换某层某位置的 residual/attention/mlp 输出”，更容易落地到你们现有工程

> 结论层面：
>
> - 没有 hook，你们最多能说“相关性差异在哪些层”。
> - 有 patching，你们才能说“哪些层/头对工具范式差异是因果必要/充分的”。

---

## E. 相关工作与方法名称（以及在 ICLR/ICML 常见用法）

下面按“你们项目最匹配”排序（只列名称与典型用法口径，方便你们写 Related Work/Method）：

### 表示相似度与对照分析

- **CKA (Centered Kernel Alignment)**：比较不同条件/模型/层的表示相似性；常用于“训练策略/微调/对齐方法改变了哪些层”。
- **SVCCA / PWCCA**：子空间级的表示对齐，常用于“表示是否在某些层发生旋转/重参数化但仍表达同一信息”。
- **Representational Similarity Analysis (RSA)**：用距离矩阵比较表示结构。

### Logit lens 系列

- **Logit Lens**：用中间层表示直接投到 vocab，看“信息何时可线性读出”。
- **Tuned Lens**：对 logit lens 做层特定校准，减少“中间层未对齐导致的读出失真”；在论文里常作为更可靠的“信息出现时刻”证据。

### 因果机制定位（你们若要“关键机制/提升工具能力”，这块最重要）

- **Activation Patching**：替换某层激活，看输出指标变化，用于验证因果贡献。
- **Causal Tracing**：更系统地在层/位置上做因果注入与回溯，常用于“事实记忆从哪层哪 token 进入输出”。
- **Path Patching**：把贡献进一步分解到注意力头、MLP、甚至路径级别。
- **EAP (Edge Attribution Patching)**：给电路“边”做归因，找到关键边集合。
- **ACDC (Automatic Circuit Discovery)**：自动化剪枝找最小电路（顶会里常用于把“patching 发现的关键模块”压缩成可解释子图）。

### 注意力可解释与归因

- **Attention Rollout / Attention Flow**：把多层注意力组合成“信息流”证据（常用于展示输入片段对输出的整体影响路径）。
- **Integrated Gradients / Grad×Input / Grad×Activation**：梯度归因（常用来补强“注意力不等于解释”的质疑）。

### 与“编辑/定位知识”相关（可作为方法论类比）

- **ROME / MEMIT**：通过定位并编辑模型内部机制来改变行为；你们如果最终目标是“增强工具使用能力”，可以类比“先定位关键层/头→再定向干预/训练”。

---

# 最建议你们的执行顺序（保证每一步都有可展示产出）

1. **先做方案 1**：canonical 语义 token 的 time×layer 热力图 + 分层曲线（立刻解决 logit lens 不可读）
2. **立刻并行做方案 2**：scaffold/DoD 把 RoPE/长度 confound 拆掉（给结论上保险）
3. **接着做方案 3**：注意力分段占比 → 找候选“工具整合头”
4. **再做方案 4**：patching 验证因果（把候选变成结论）
5. **最后做方案 5**：按 G10/G01 分组，把“差异”与“收益”绑定（论文叙事闭环）

---

如果你愿意，我也可以在你们现有三张均值曲线（CKA/KL/logit lens）的基础上，帮你们把“下一张最该出的图”具体化成：

- 该选哪些 canonical 语义 token（数量、类别、过滤规则）
- heatmap 的轴与聚合方式怎么定
- 以及你们现在 CKA 在 30+ 层下滑、KL 在最后层尖峰的“最可能解释路径”应该如何用方案 2+4 去验证。
