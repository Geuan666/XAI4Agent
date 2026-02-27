下面我会按你们现在要产出的 **三张图** 来把“teacher-forcing 差分”这条主线进一步落到**可操作的定义、计算口径、以及分析流程**上（不含代码）。同时我会补上**顶会论文里可对齐的先例**，让你们的方法在组会/论文里“有据可依”，而不是看起来像拍脑袋。

---

# 1) 先回答：什么是 layer×token-index 的 ΔNLL（或 JS）？

你们现在已经能构造 pair（agentic / assisted）并在 teacher forcing 下对“同一段 canonical continuation”做前向传播拿到所有 hidden states。接下来要做的事情，本质是把“每一层、每一个生成步”都变成一个可比较的标量指标。

## 1.1 teacher forcing 下的 token-index 是什么

设：

* prefix（真实前缀输入）记为 (x_{1:m})
* continuation（canonical 函数体 token 序列）记为 (y_{1:T})

把它们拼起来得到完整序列：
[
s = [x_{1:m}, y_{1:T}]
]

在标准自回归 LM 的 teacher forcing 评估里：

* 模型在位置 (m+t-1) 产生的 logits，对应预测 **下一个 token** (y_t)

也就是说：
**token-index (t)** 指的就是 continuation 的第 (t) 个 token（你们最终会把它改成“第 (t) 个语义 token”，下面会讲）。

## 1.2 ΔNLL 的定义（按 token、按层）

对某个范式 (c\in{\text{agentic},\text{assisted}})，在某一层 (l)，对 continuation 的第 (t) 个 token：

1. 通过该层的“logit lens readout”（你们已有实现）把该层在位置 (m+t-1) 的隐状态解码成词表分布：
   [
   p^{(c)}*{l,t}(\cdot) = \text{softmax}(\text{LogitLens}(h^{(c)}*{l,,m+t-1}))
   ]

2. 取 canonical 的真实下一个 token (y_t) 的概率，定义 NLL：
   [
   \text{NLL}^{(c)}*{l,t} = -\log p^{(c)}*{l,t}(y_t)
   ]

于是你们要画的热力图值就是：
[
\Delta\text{NLL}*{l,t} = \text{NLL}^{(\text{agentic})}*{l,t} - \text{NLL}^{(\text{assisted})}_{l,t}
]

解释非常直观：

* (\Delta\text{NLL}_{l,t} < 0)：agentic 在该层该步对“正确 token”更自信（更像“工具信息帮助了预测”）
* (\Delta\text{NLL}_{l,t} > 0)：agentic 更不自信（可能“工具段干扰/分心/规划不同”）

> 这类“用 logprob / negative logprob / logit difference 作为定位与解释指标”的口径，在 MI 顶会工作里是非常常见的：
>
> * ROME（NeurIPS 2022）明确用“输出 token 概率变化”来做 causal tracing 指标。
> * IOI 电路（ICLR 2023）用 logit difference，并指出它等价于训练时 loss 的差。
> * ACDC（NeurIPS 2023）把不同任务的指标列成表，包括 logit difference、probability difference、negative log-probability。 ([NeurIPS Proceedings][1])
> * ICLR 2024 的 activation patching best practices 也系统讨论了 probability / logit difference / KL 作为评价指标。

所以你们做 ΔNLL（或等价的 Δlogprob / logit diff）是“有迹可循”的。

## 1.3 JS（Jensen–Shannon divergence）的定义（按 token、按层）

ΔNLL是“只看正确 token 的概率”。JS 是“看整个位分布是否变了”，非常适合你们现在遇到的 **KL 最后层尖峰**问题（KL 对尾部分布/极小概率很敏感，容易被尖峰主导；JS 有界更稳）。

对同一层、同一步，你们有两个分布：

* (p^{(\text{agentic})}_{l,t})
* (p^{(\text{assisted})}_{l,t})

定义：
[
\text{JS}_{l,t} = \frac{1}{2}\text{KL}(p_a \Vert m) + \frac{1}{2}\text{KL}(p_b \Vert m), \quad m=\frac{p_a+p_b}{2}
]

然后热力图画：

* ( \text{JS}_{l,t} )（表示“差异强度”，不带方向）
  或画
* (\Delta\text{NLL}_{l,t})（表示“差异方向 + 强度”，更像“agentic 有无帮助”）

**推荐**：热力图主用 ΔNLL（有方向），曲线主用 JS（稳且不被尖峰带跑）。

---

# 2) 你们要的第 1 张图：canonical 语义 token 的 layer×token-index ΔNLL/JS 热力图 —— 具体怎么做？

你问“拿到 hidden state 接下来怎么做”，我给一个**从单样本 → 全数据**可逐层推进的方案。

## 2.1 Step 0：保证 prefix 真的是“真实运行前缀”

你们已经强调：不能猜模板，必须用 trace 重建。

* **agentic**：用 `agentic_output.json` 的 `message_trace` + 真实工具 schema（来自你们 `Agent/agent.py` 和 trace 实际出现文本）重建“read_file 后那次前向”的 messages
* **assisted**：用 `assisted_run.py` 当时拼 prompt 的同逻辑/或 output.json 里保存的 prompt

并且用 server1 保存的：

* `token_texts`
* `token_spans`
* `prompt_offsets`

来做一致性校验，确认“分析输入 == 当时前向输入”。

> 这一步是你们后面所有图的信誉地基。

## 2.2 Step 1：构造 continuation 的“语义 token”序列（解决空白 token 不直观）

你们现在 logit lens 目标 token 经常落在缩进空白，这是最需要修的口径。

### 最低成本且非常有效的语义 token 过滤规则

对 continuation 的 token 列表 (y_{1:T})，建立一个索引集合 (S)：

* 丢弃：解码后 `strip()==""` 的 token（空白/缩进/纯换行）
* 保留：其他 token

得到语义 token 序列 (y_{S(1)}, y_{S(2)}, \dots)，把热力图横轴改成 **semantic-index**（第几个语义 token）而不是原始 token-index。

> 为什么不直接用 AST？
> 你们当然可以后面升级到 AST span，但第一版“strip 过滤”就能把 80% 的可解释性救回来，而且完全贴合你们工程最小增量目标。

### 进一步更精细（你们说会做“更精细分类”）

在 strip 过滤后，再做轻量分类（用于标注热力图上关键阶段）：

* 结构 token：`if/for/while/return/try/except/with` 等
* 标识符 token：变量名、函数名、属性
* 字面量/运算符：数字、字符串、`==`, `+` 等

这可以让你们回答：“差异发生在结构规划 vs 变量拷贝”。

## 2.3 Step 2：对每个 layer×semantic-index 计算 ΔNLL 或 JS

假设你们一次 teacher forcing 前向拿到了每层 hidden states（你们说“拿到了所有 hidden state”，那就足够了）：

对每个层 (l)：

* 对每个语义 token index (k)（对应原 token (t=S(k))）：

  * 取位置 (m+t-1) 的 hidden（预测 (y_t)）
  * logit lens 解码出 (p_{l,t})
  * 算 NLL 或 JS
  * agentic 与 assisted 做差/做 JS

得到矩阵：

* (\Delta\text{NLL}*{l,k}) 或 (\text{JS}*{l,k})

## 2.4 Step 3：热力图“长短不一”的对齐（从单样本扩展到 164 样本时必须解决）

单样本不需要这一步。
但你们做全数据均值热力图时，HumanEval 每题 continuation token 数不同。常见三种做法（都能写进论文）：

### 做法 A（最建议，解释“阶段”最自然）：按生成进度归一化到固定 bins

* 把每题语义 token 序列长度 (K_i) 映射到 ([0,1]) 的进度
* 固定 50 或 100 个进度 bin
* 把每个 token 的 ΔNLL/JS 聚合到对应 bin（平均/中位数）
* 于是全数据能得到统一的 layer×progress 热力图

优点：最贴你们“差异发生在生成哪个阶段”的目标。

### 做法 B：只看前 K 个语义 token（例如前 128）

* 超过 K 的截断
* 少于 K 的留空（NaN）或 padding 后 mask

优点：实现简单；缺点：只能说“前段”。

### 做法 C：按“结构锚点”对齐（return/first if/first loop）

* 对每题找到锚点 token 的位置
* 把横轴变成“锚点对齐后的相对距离”
  优点：很能讲故事；缺点：工程上要多做一点规则。

---

# 3) 你们要的第 2 张图：每层 ΔNLL/JS 曲线 + bootstrap 95% CI —— 怎么把“最后层尖峰”从根上解决？

你们现在 mean KL 曲线的尖峰问题，本质来自两点：

1. KL 对极小概率很敏感（最后层常把某些 token 概率压到接近 0，导致 KL 爆）
2. 你们可能是在“最后 token/少数 token”上算，统计不稳

**替换策略**：用 **JS** 或 “ΔNLL（语义 token 平均）”做 per-layer curve，并做 bootstrap CI。

## 3.1 每层指标怎么定义（建议统一写法）

对每个样本 (i)，你已经有 layer×semantic-token 的矩阵（来自第 1 张图的中间结果）。

对每层 (l)，定义该样本的标量：

* (\Delta\text{NLL}*{i,l} = \text{mean}*{k \in \text{semantic}} \Delta\text{NLL}_{i,l,k})
  或
* (\text{JS}*{i,l} = \text{mean}*{k \in \text{semantic}} \text{JS}_{i,l,k})

然后跨样本取均值曲线：

* (\overline{\Delta\text{NLL}}_l = \text{mean}*i \Delta\text{NLL}*{i,l})
* (\overline{\text{JS}}_l = \text{mean}*i \text{JS}*{i,l})

> 你们会发现：
>
> * JS 曲线通常平滑，最后层不会像 KL 一样爆炸
> * ΔNLL 曲线能告诉你“agentic 整体更好/更差在哪些层”

这种“用 probability/logit difference/negative logprob 做指标，并讨论哪种指标更稳健”的口径，在 ICLR 2024 activation patching best practices 里就是专门讨论点。

## 3.2 bootstrap 95% CI 怎么做（建议按任务维度 resample）

你们有 164 题。最标准、最容易解释的 bootstrap：

* bootstrap 单位：**题目（sample）**
* 每次重采样：从 164 题中有放回抽 164 题
* 对每次重采样，重新计算 (\overline{\Delta\text{NLL}}_l) 或 (\overline{\text{JS}}_l)
* 重复 B 次（1000 或 5000）
* 对每层 (l)，取 2.5% 与 97.5% 分位数作为 CI

## 3.3 你们要做“四象限分组”时的 CI（推荐分层 bootstrap）

你们说会按通过/不通过分四象限（G11/G10/G01/G00）。建议：

* 先在每组内做 bootstrap（只在组内重采样），得到组内 CI
* 如果要做“组间差”（例如 G10 的曲线 - G01 的曲线），同一次 bootstrap 里分别在两组内采样再相减，得到差值 CI

这样你们能非常强地回答：

* “差异是不是主要集中在带来收益的 G10 组，而不是全体平均的幻觉？”

---

# 4) 你们要的第 3 张图：segment attention mass 曲线（对 tool-response 的注意力占比）—— 取什么 attention？怎么分析？

你说“先不考虑算力/显存/存储限制，科研导向”。那我就按“能做最强证据链”的方式给方案，并且尽量让它跟你们第 1 张热力图能形成联动（同样基于 teacher forcing，避免输出不一致导致不可比）。

## 4.1 取什么 attention：建议用 “teacher forcing 下、预测语义 token 的 query 位置”的 attention

你们的目标是证明：**模型在生成函数体时确实读取了 tool-response 段**，并定位“在哪些层读”。

因此 attention 的定义要跟你们的 ΔNLL 对齐：

* 对 continuation 的目标 token (y_t)（或语义 token (k)），预测它的 query 位置是 (q = m+t-1)
* 在层 (l)、头 (h)，注意力权重矩阵记作：
  [
  A^{(c)}_{l,h}(q,\cdot)
  ]
  这是一个对所有可见 key 位置（prefix + 之前的 continuation）归一化后的分布（softmax 后）。

> 这种“在特定 token 位置用概率/attention 做分析并定位层”的思路，在 causal tracing / activation patching 系列工作里非常常见：他们就是围绕“某个位置输出 token 的概率/logit”去问哪些内部组件重要。

## 4.2 segment attention mass 的核心定义

你们已经能用 `token_spans + prompt_offsets` 定位 segment（这是对的，不要硬切绝对位置）。

设 tool-response segment 对应的 token index 集合为 (R)。

那么对某层某头、在预测 (y_t) 的那个 query 位置 (q)，定义：

[
\text{Mass}^{(c)}*{l,h,t \rightarrow R} = \sum*{i \in R} A^{(c)}_{l,h}(q,i)
]

性质：

* 这是一个 0~1 之间的值
* 越大表示“这个预测步更依赖从 tool-response 段取信息（至少从 attention 路径上看）”

然后你们可以做不同粒度的聚合来画曲线：

### 最核心、最像你们要的“曲线图”

对每层 (l)，先对头平均，再对语义 token 平均：

[
\text{Mass}^{(c)}*l
= \mathbb{E}*{t \in \text{semantic}}\left[\mathbb{E}*{h}\left[\text{Mass}^{(c)}*{l,h,t \rightarrow R}\right]\right]
]

画出来就是：**layer → tool-response attention mass 曲线**

这张图你们想表达的结论一般是：

* agentic 在某些中后层（例如 L20-L35）对 tool-response mass 明显升高
* 或者存在一段“读取窗口”（某些层 mass 峰值很明显）

## 4.3 “怎么分析 attention”——给你一条从弱到强的证据链（科研导向）

如果只画一条 mass 曲线，组会上容易被问：
“attention 是不是虚假的？是不是只是形式上看了但没用？”

所以建议你们把 attention 分析做成 3 个层级，每个层级都能用现成数据做：

### Level 1（描述性证据）：mass 曲线 + 分段对照

在 agentic 内部做“segment mass 分解”，同时画：

* ToolResponse mass
* Spec mass（函数签名/docstring）
* Prior-output mass（已经生成的函数体 tokens）
* ToolCall/schema/instruction mass（若存在）

你会得到一个很直观的结构图：模型到底在“看资料”还是“看自己刚写的内容”。

### Level 2（与行为改进挂钩）：把 mass 与 ΔNLL 热力图对齐做相关

对同一个 layer×token-index 网格：

* 有 (\Delta\text{NLL}_{l,t})
* 也有 (\Delta\text{Mass}*{l,t\rightarrow R}) 或 agentic 的 (\text{Mass}*{l,t\rightarrow R})

你们可以回答更强的问题：

* “当 tool-response mass 升高的那些 token 步，ΔNLL 是否更偏负（预测更好）？”
* 尤其在 G10（agentic 带来收益）组，这个相关性应该更强

这会让“读 tool”从直觉变成定量关系。

### Level 3（更像因果的证据，但仍不需要 hook）：tool-response 内容干预对比

你们之前也提过 B1（neutral response），这里给出最直接的科研策略：

保持 agentic 流程完全不变（仍只允许 read_file），但 tool response 内容换成：

* 原始 main.py（真实）
* 重复 spec（信息不增量）
* 错题 main.py（长度匹配但信息错误）

然后比较：

* tool-response attention mass 是否仍高
* ΔNLL 是否仍改善
* pass rate 是否下降

这条链路在机制论文里非常好写：
“当工具信息被移除/污染时，对工具段的注意力与性能提升同步消失”，比单看 attention 更有说服力。

---

# 5) 你提的“padding 控制长度变量”——我建议你们怎么把它做严谨（避免做成“看似控制但其实没控制”）

你们的想法是对的：prompt 长度差会带来 RoPE 位置差，污染中后层对齐。
但要注意一个工程陷阱：

> **很多 HuggingFace 实现会根据 attention_mask 生成 position_ids（非 pad 从 0 开始），这会导致“你虽然 pad 了，但真实 position 并没有平移”。**

所以如果你们要用“pad + attention_mask=0”来对齐 position，建议把方案写成两步：

## 5.1 主分析建议保留两套口径（论文更稳）

* **As-run 模式**：不做 pad（完全真实运行输入），这反映真实 agentic vs assisted 行为差
* **Pos-matched 控制模式**：对 assisted 做左 padding + 明确检查/设置 position_ids，使 continuation 的绝对位置与 agentic 一致

这样你们能回答质疑：

* “这差异是不是只是长度造成？”
  你可以说：在 pos-matched 控制下差异仍存在/显著减弱（分别对应“真实工具效应”与“长度混杂占比”）。

## 5.2 Pos-matched 模式必须做的 sanity check（写进实验步骤）

对同一条输入：

* 抽查 continuation 的第一个语义 token (y_{S(1)}) 在 agentic 与 assisted-posmatched 中的 **position id 是否相同**
* 如果 server1 没存 position_ids，建议加一个“记录 position_ids/rope index”的保存项（这是非常小的工程改动，但能避免无效控制）

---

# 6) 顶会论文里哪些工作“有 teacher-forcing 的分析思路”？（你要的依据）

你们的做法可以拆成两个部分：

1. **teacher forcing 下计算 token-level probability / NLL / logit difference**
2. 把这个指标扩展到 **layer-wise**（logit lens / tuned lens 风格）

对应的顶会/主流 MI 先例：

## 6.1 NeurIPS 2022 — ROME / causal tracing（用输出概率作为关键指标）

ROME 这类工作就是围绕“给定 prompt，模型对目标 token 的概率 P(o) 如何变化”，并用此做因果定位与验证。

## 6.2 ICLR 2023 — IOI circuit（logit difference = loss difference）

IOI 电路论文明确用 logit difference 衡量任务表现，并指出它对应训练时 loss 的差异（非常“teacher forcing”）。

## 6.3 NeurIPS 2023 — ACDC（把“指标”系统化：logit diff / prob diff / negative logprob）

ACDC 系统总结了 MI 工作流：选任务/数据集/指标，然后用 patching 找电路；其中指标里就包含 negative log-probability 等 teacher-forcing 下的 next-token 指标。 ([NeurIPS Proceedings][1])

## 6.4 ICLR 2024 — Activation patching best practices（系统讨论 probability/logit diff/KL 指标）

这篇基本可以作为你们“为什么用 JS/ΔNLL 替代 KL”的方法论依据：它专门讨论不同指标的性质与 pitfalls，并给出推荐。

## 6.5 ICLR 2023 — MEMIT（语言模型定义就是条件分布，分析也围绕目标 token 概率）

MEMIT 作为 ICLR 2023 会议论文，在方法预备里就把 LM 写成条件 token 分布 (P(x[t]\mid x[1..E]))，属于你们 teacher-forcing 分析的理论底座。

## 6.6 “层级解码成分布”的方法论文：Tuned Lens（logit lens 的稳健版）

Tuned Lens 明确就是“把每层 hidden state 解码成词表分布、观察 layer-by-layer 的预测轨迹”，非常贴合你们要画的 layer×token heatmap，只是它用训练过的 probe 来降低 logit lens 的 brittle。 ([arXiv][2])

> 你们不一定要立刻上 tuned lens（因为要训练 probe），但可以把它写进 related work / future work，解释“我们用 vanilla logit lens readout，但可用 tuned lens 提升稳健性”。

---

# 7) 给你一份“精细化后的三图产出方案”（按你们的推进节奏：单样本 → 分组 → 全体）

下面这份就是你们接下来可以直接照着执行的 SOP（无代码版）。

---

## 图 1：canonical 语义 token 的 layer×token-index ΔNLL（或 JS）热力图

### 输入准备

1. 对单个样本 i：

* agentic：从 `agentic_output.json` 取该题的 `message_trace`，重建 read_file 后那次前向的真实 prefix
* assisted：取对应题目的真实 prefix

1. continuation：用 `canonical_solution` 提取函数体（你们已有），tokenize 得到 (y_{1:T})
2. 语义 token 集 (S)：过滤 `strip()==""` 的 token

### 前向与对齐

1. teacher forcing 前向：分别对 agentic 与 assisted 跑 full sequence logits/hidden（至少要能做 logit lens）
2. 可选控制：做 assisted 的 pos-matched（左 padding + position sanity check）

### 指标矩阵计算

1. 对每层 (l)，对每个语义 token (k)（对应原 token (t=S(k))）：

* 取预测位置 (q=m+t-1)
* logit lens 解码得到分布 (p_{l,t})
* 计算：

  * (\Delta\text{NLL}_{l,k})（有方向）
  * 或 (\text{JS}_{l,k})（无方向但稳）

### 可视化建议

1. 单样本：直接画 layer×k 热力图
2. 标注：在横轴挑选 5~10 个 token 显示其解码文本（例如关键 `return`/变量名）
3. 之后扩展到组/全体：用 progress bin 或前 K 语义 token 对齐（建议优先 progress bin）

---

## 图 2：每层 ΔNLL/JS 曲线 + bootstrap 95% CI（替换 mean KL）

### per-sample 标量化

1. 对每题 i、每层 l：

* (\Delta\text{NLL}_{i,l} = \text{mean}*k \Delta\text{NLL}*{i,l,k})
* 或 (\text{JS}_{i,l} = \text{mean}*k \text{JS}*{i,l,k})

### 全体曲线与 CI

1. 全体均值曲线：对 i 求均值
2. bootstrap（按题目重采样）得到每层 95% CI
3. 分组（四象限）同理，并推荐做 stratified bootstrap

### 你们能写进结论的话术模板

* “我们用 JS/ΔNLL 替代 KL，避免 KL 在最后层被极小概率尖峰主导（这类指标选择问题在 activation patching 文献中也被系统讨论）。”
* “差异集中在第 L_a–L_b 层，并在 G10 组最显著。”

---

## 图 3：segment attention mass 曲线（对 tool-response 的注意力占比）

### segment 对齐（你们已有能力）

1. 用 `token_spans + prompt_offsets` 标出 tool-response segment 的 token 集合 (R)

### attention 取法

1. 在 teacher forcing 下，对每个目标语义 token (y_t)，取其预测 query 位置 (q=m+t-1)
2. 对每层、每头取 attention 权重向量 (A_{l,h}(q,\cdot))

### mass 计算与聚合

1. 计算 (\text{Mass}*{l,h,t\rightarrow R}=\sum*{i\in R}A_{l,h}(q,i))
2. 聚合：

* per-layer curve：对 head、对语义 token 平均
* 可选：layer×token-index mass 热力图（可和 ΔNLL 热力图对照）

### 强化解释（强烈建议）

1. 在 G10 vs G01 分组上分别画 mass 曲线：看“读 tool”是否只在带来收益的组更强
2. 做 tool-response 内容干预（重复 spec / 错题 main.py），验证 mass 与 ΔNLL/性能同步变化（更像因果链）

---

如果你愿意，我下一步可以把“热力图的 token-index 对齐策略（progress bin / anchor 对齐）”再进一步落到**你们 HumanEval 代码任务**的具体锚点上（比如按“首个 return / 首个控制流 / 最后 return”对齐），这样更容易写出“生成阶段”的解释叙事。

[1]: https://proceedings.neurips.cc/paper_files/paper/2023/file/34e1dbe95d34d7ebaf99b9bcaeb5b2be-Paper-Conference.pdf "Towards Automated Circuit Discovery for Mechanistic Interpretability"
[2]: https://arxiv.org/abs/2303.08112?utm_source=chatgpt.com "Eliciting Latent Predictions from Transformers with the Tuned Lens"

下面我只围绕你问的 **“Step 2 怎么具体算 ΔNLL/JS（layer×token-index）”** 来讲清楚，并把“要不要跑 t 次前向”这个误解纠正掉；最后再给你补一波**顶会论文里确实用 teacher-forcing / token-level logprob 这种口径**做机制分析的依据（我会带引用，方便你们写 related work / method justification）。

---

## 1) 你说的理解基本对，但 **不需要跑 t 次前向传播**

你理解的热力图语义是对的：**每个格子就是“某一层 l，在预测 canonical 第 t 个 token 时，对正确 token 的 NLL（或与另一范式的差）”**。
从图像上看，你可以把：

* 横轴 = layer（第几层）
* 纵轴 = token-index（canonical continuation 里的第几个 token/语义 token）

或者反过来也行（只是画图方向不同，信息等价）。

### 关键纠偏：为什么不需要 t 次 forward？

因为在 **teacher forcing** 下，你把 `prefix + canonical_continuation` 作为一个**完整序列**喂给 decoder-only LM 做一次 forward：

* Transformer 计算是并行的
* causal mask 保证每个位置只能看见左侧历史（不会“偷看未来 token”）

所以**一次 forward 会同时给出所有位置的 next-token 预测分布**。
也就是说：对一个样本、一个范式（agentic 或 assisted），你只要 **1 次 forward** 就拿到了所有 token 步的 logits/hidden。
最终只需要 **2 次 forward**（agentic 一次 + assisted 一次）就能算完整的 layer×token 热力图。

> 你如果用的是“只返回 last token logits/hidden”的接口（像逐 token generate），那才会逼着你做 t 次 forward。
> 但你现在的目标是“拿到所有 hidden states”，那就应该走 **full-seq forward** 的 teacher forcing 方式。

---

## 2) layer×token-index 的 ΔNLL：到底怎么算（最核心）

这里我用最严谨、不会 off-by-one 的方式描述。你们实现时只要把 “cont_start” 标准化，就不容易错。

### 2.1 先定义序列与索引

对某个范式 (c \in {\text{agentic}, \text{assisted}})：

* prefix token 序列：(x_{0:m-1})
* canonical continuation token 序列：(y_{0:T-1})
* 拼接得到 full input：
  [
  s = [x_0,\dots,x_{m-1}, y_0,\dots,y_{T-1}]
  ]
* continuation 的起点 token 下标：
  [
  \text{cont_start} = m
  ]

> 你们工程上 cont_start 不一定就是 m（因为你们可能把 continuation 接到某个 message 后面，或者 pad/模板有特殊 token）。
> 所以实际实现建议 **直接用 token_spans/prompt_offsets 或 “拼接前后的 token 长度差”** 得到 cont_start，并保存下来。

### 2.2 “预测 y_t” 应该取哪个位置的 logits？

在标准 causal LM（GPT 类）里：

* 输出 logits 的位置 index (i) 对应预测 **下一个 token** (s_{i+1})

因此 canonical 的第 (t) 个 token（(y_t = s_{\text{cont_start}+t})）对应的预测位置是：

[
\text{pos}(t) = (\text{cont_start}+t) - 1
]

解释一下：

* (t=0)：预测 (y_0) 用的是 prefix 最后一个 token 的位置 logits，也就是 (\text{cont_start}-1)
* (t=T-1)：预测最后一个 (y_{T-1}) 用的是 (\text{cont_start}+T-2)

这就是你算 token-level NLL 时最容易错的地方：**永远是“目标 token 的前一位 logits”**。

### 2.3 为什么你现在能做 “每层” NLL？——靠 logit lens

如果你要做 (\text{NLL}^{(c)}_{l,t})（带 layer 维度），就不是用 final logits，而是用 **logit lens**：把第 (l) 层该位置的 hidden state 解码成一个词表分布。

你已经拿到了所有层 hidden states：

* (h^{(c)}_{l, i})：范式 c、层 l、位置 i 的 hidden 向量

logit lens 的核心就是：用模型本身最后的 unembed head（通常是 “final norm + lm_head”）把 (h_{l,i}) 映射到 vocab logits：

[
z^{(c)}*{l,i} = \text{LMHead}(\text{FinalNorm}(h^{(c)}*{l,i}))
]

> 这点很重要：很多模型（包含 GPT-2 等）在 unembed 前有 final layernorm；IOI 论文也明确描述了 “forward pass 末尾会有 layer norm 再接 unembed”。([OpenReview][1])
> 所以你们做 per-layer logit lens 时，最好复用同一条 norm+head 路径，否则中间层 logits 尺度会很怪（“logit lens brittle”问题，tuned lens 就是专门为这个提出的）。

然后 softmax 得到分布：
[
p^{(c)}*{l,i}(\cdot)=\text{softmax}(z^{(c)}*{l,i})
]

### 2.4 计算 per-layer per-token NLL

目标 token 是 (y_t)，预测位置是 (i=\text{pos}(t))。

[
\text{NLL}^{(c)}*{l,t} = -\log p^{(c)}*{l,\text{pos}(t)}(y_t)
]

最后做差分：
[
\Delta \text{NLL}*{l,t} = \text{NLL}^{(\text{agentic})}*{l,t} - \text{NLL}^{(\text{assisted})}_{l,t}
]

这就是热力图的一个格子。

### 2.5 一次 forward 里怎么“批量化”算完整热力图？

你不需要循环 t 做 forward，但你确实要在 **后处理阶段**循环/向量化地取出：

* 所有层 (l=1..L)
* 所有 token 步 (t\in S)（语义 token 集合 S）

把 (\text{pos}(t)) 这一列 logits/hidden 抽出来算就行。

---

## 3) layer×token-index 的 JS：怎么画、怎么算？（同样不需要 t 次 forward）

你对 JS 的理解需要稍微调整一下：
**JS 不是“标准答案的影响”**，而是“两个范式在该步的 next-token 整体分布差异有多大”。

### 3.1 per-layer per-token 的 JS 定义

还是对每个层 (l)、每个 token 步 (t)：

* 先得到两个分布：
  [
  p_a = p^{(\text{agentic})}*{l,\text{pos}(t)},\quad
  p_b = p^{(\text{assisted})}*{l,\text{pos}(t)}
  ]
* 混合分布：
  [
  m=\frac{p_a+p_b}{2}
  ]
* JS：
  [
  \text{JS}_{l,t}=\frac12\text{KL}(p_a|m)+\frac12\text{KL}(p_b|m)
  ]

画热力图就画 (\text{JS}_{l,t})（或对语义 token 的 (t)）即可。

### 3.2 你问“是不是要看最后一层隐藏层对标准答案的影响？”

* 如果你只关心 final layer，那就用 final layer logits 来算 (\text{JS}_{L,t})（这会变成一条随 t 变化的曲线/热力条）
* 但你们要的是 **layer×token**，所以建议跟 ΔNLL 一样：用 logit lens 在每层解码得到 (p_{l,i}) 后再算 JS（这样才有 layer 维度）

### 3.3 JS 这张图能回答什么？

* JS 大：说明 agentic/assisted 在该步的“候选 token 分布”差异明显（即决策空间变化大）
* JS 小：说明两者决策非常接近
  但它不告诉你“谁更接近标准答案”，那个要看 **ΔNLL（有方向）**。

---

## 4) 你要的“有依据”的 teacher-forcing 思路：再补一批顶会论文证据

你担心“我们是不是乱做”，这担心很合理。下面是我再检索后，**跟你们的做法最直接同构**的证据链（都可以写进 method/related work）：

### 4.1 ICLR 2023 — IOI 电路论文：logit difference ≈ training loss difference（teacher-forcing 口径）

IOI 论文明确说他们用 **logit difference** 作为任务指标，并指出这等价于训练时的 loss 差异：
“这是模型在训练时如果 IO 是正确 vs 如果 S 是正确会收到的 loss 差异”。([OpenReview][1])
这几乎就是在告诉你：用 teacher-forcing 下的 token-level logit / loss 作为 mechanistic metric 是正统做法。

### 4.2 NeurIPS 2022 — ROME：直接做 “layer-and-token combination” 网格评估

ROME 的 Figure 5 明确写：他们把结果 **在每个 layer×token 组合上做 benchmark**，并据此定位关键层段（mid-layer MLP、特定 token 位置）。
这跟你们要做的 layer×token ΔNLL/JS 热力图在“结构上”高度一致。

### 4.3 ICLR 2024 — Activation patching best practices：讨论 metric（logit diff vs probability）+ 展示跨 layer & token position 的图

这篇 ICLR 2024 系统讨论了 activation patching 的方法细节，其中包括：

* 不同 **evaluation metric（logit difference vs probability）** 会导致不同 localization 结论
* 并且他们给出跨 **layers & token positions** 的 patching 图（例如 Figure 18）。
  你们用 ΔNLL/JS 替换 KL、并强调“不要让最后层尖峰统治”，也完全能在这篇里找到方法论支撑。

### 4.4 NeurIPS 2023 — ACDC：把“选指标 + 选数据 + 做 patching/对比”系统化，并讨论 logit diff / KL 的问题

ACDC 明确把 mechanistic interpretability 的流程写成：先选 metric 和 dataset，再做 activation patching / circuit discovery，并且专门讨论 **logit difference vs KL** 这类指标问题。
你们现在要做的“换指标口径（JS/ΔNLL）来避免 KL 尖峰”可以直接引用这条思路。

### 4.5 NeurIPS 2020 — Causal Mediation Analysis：通过干预看 predicted probability 的变化

Vig 等人的 causal mediation analysis（NeurIPS 2020）是更早一代“干预—看输出概率变化”的机制解释框架。
虽然它不一定是你们这种 layer×token heatmap，但它提供了“用输出概率/分布变化作为因果指标”的经典先例。

### 4.6（补充但不是顶会）Tuned Lens：直接画 “每个 layer×token 的预测”

Tuned Lens 的论文里有非常直观的图：**每个格子就是某层某 token index 的 top-1 预测**。
这几乎就是你们想做的热力图（只不过你们画的是 ΔNLL/JS，而他们画的是 top-1 token）。
它不是顶会定稿论文（至少我没在检索里看到明确 conference proceedings），但在机制分析圈子里是非常主流的“层级预测轨迹”依据。

---

## 5) 给你一个“最精炼的计算流程 checklist”（就对应你现在卡住的 Step 2）

对每个样本 i、每个范式 c（agentic/assisted）只做一次 forward：

1. 得到 `hidden_states[l, seq_pos]`（所有层、所有位置）
2. 对每层 l：用 logit lens 解码得到 `logits_l[seq_pos, vocab]`
3. continuation 的每个 token (t)：

   * `seq_pos = cont_start + t - 1`
   * 目标 token id = `y[t]`
   * NLL：(-\log \text{softmax}[logits_l[seq_pos]](y[t)])
   * JS：用两个范式在该 `(l, seq_pos)` 的分布算 JS
4. 组装出矩阵：

   * ΔNLL([l,t])（有方向）
   * JS([l,t])（无方向）

**总 forward 次数：**每样本 2 次（agentic + assisted），不是 2T 次。

---

如果你愿意，我还能把上面 **pos(t) 的 off-by-one** 再用你们实际数据的例子（比如某条 HumanEval 的 token_texts）具体走一遍：告诉你 cont_start 应该落在哪、为什么 (t=0) 对应 cont_start-1。这个能大幅降低实现时“热力图全错位”的风险。

