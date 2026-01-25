# Engram: 基于可扩展查表的条件记忆架构（代码分析版）

**论文标题**：Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
**论文链接**：[GitHub](https://github.com/deepseek-ai/Engram)
**作者团队**：北京大学 & DeepSeek-AI

> 本文为 **academic 风格 + 代码分析** 示例。在完整论文解读基础上，结合开源代码讲解实现细节。

---

## 摘要

本文提出 Engram，一种基于条件记忆（Conditional Memory）的新型稀疏架构。该方法将 N-gram 查表机制与 Transformer 主干网络相结合，通过可扩展的哈希检索实现 O(1) 复杂度的静态知识获取，同时保留 Transformer 的动态推理能力。实验表明，在等参数、等 FLOPs 条件下，Engram-27B 在知识任务（MMLU +3.4）、推理任务（BBH +5.0）和长上下文任务（NIAH 84.2→97.0）上均显著优于 MoE 基线。

## 1. 研究背景与动机

### 1.1 当前大语言模型的局限性

现有大语言模型（LLMs）将所有知识编码于密集参数中，推理时需激活完整的计算路径。这种设计存在以下核心问题：

**计算效率问题**：静态知识（实体名称、固定搭配、事实性信息）与动态推理（逻辑推演、数学证明）共享相同的计算资源。获取"中国的首都是北京"这一简单事实，需要与复杂推理任务相同的计算开销。

**知识更新困难**：模型的知识完全编码于参数中，修改特定知识点需要重新训练整个模型，缺乏模块化的知识管理机制。

**内存效率低下**：大量参数用于存储可查表的静态模式，而非真正需要参数化的推理能力。

### 1.2 N-gram 模型的启示

传统 N-gram 语言模型通过统计 n 个连续 token 的共现概率进行预测。其优势在于 O(1) 的查表复杂度，但受限于组合爆炸问题——n 增大时，可能的 N-gram 数量呈指数增长（V^n，其中 V 为词汇表大小）。

Engram 的核心思想是：**将 N-gram 的高效查表与 Transformer 的上下文建模相结合**，用查表处理静态模式，用计算处理动态推理。

### 1.3 项目结构概览

```
Engram/
├── engram/
│   ├── layers/
│   │   ├── memory.py      # 核心：条件记忆模块
│   │   ├── attention.py   # 注意力层
│   │   └── mlp.py         # FFN/MoE 层
│   ├── model.py           # 模型定义
│   └── config.py          # 配置
├── scripts/
│   ├── train.py           # 训练脚本
│   └── inference.py       # 推理脚本
└── configs/
    └── engram_27b.yaml    # 27B 模型配置
```

## 2. Engram 架构设计

### 2.1 整体架构

Engram 在标准 Transformer 的特定层引入条件记忆模块（Conditional Memory Module），与 MoE 专家并行部署：

![Engram 架构图](../style_comparison/images/853aef65961c6b9429369ceb08f0accc4a3021ce66d5ad2e0df9aeb3f12d479e.jpg)

架构包含三个核心组件：N-gram 哈希检索、上下文门控机制、残差融合。

### 2.2 N-gram 哈希检索

对于位置 t 的输入 token，提取其前 k 个 token 组成 N-gram（实验中 k=2 或 k=3）。通过多头哈希函数将 N-gram 映射至嵌入表：

$$e_t = \sum_{h=1}^{H} E_h[\text{hash}_h(x_{t-k:t})]$$

其中 E_h 为第 h 个嵌入表，hash_h 为对应的哈希函数。多头设计可降低哈希冲突的影响。检索复杂度为 O(1)，与序列长度无关。

#### 💻 代码实现

**论文描述**：
> "We use multi-head hashing to map N-grams to embedding tables, achieving O(1) lookup complexity."

**代码实现** (`engram/layers/memory.py`):

```python
class EngramMemory(nn.Module):
    """条件记忆模块的核心实现"""

    def __init__(
        self,
        num_heads: int = 4,
        embed_dim: int = 4096,
        table_size: int = 100_000_000,  # 1 亿条记忆
        ngram_size: int = 3,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ngram_size = ngram_size

        # 多头哈希权重（可学习）
        self.hash_weights = nn.ParameterList([
            nn.Parameter(torch.randn(ngram_size, embed_dim))
            for _ in range(num_heads)
        ])

        # 每个头独立的嵌入表
        self.embedding_tables = nn.ParameterList([
            nn.Parameter(torch.randn(table_size, self.head_dim))
            for _ in range(num_heads)
        ])

    def compute_hash(self, ngram_embeds: Tensor, head_idx: int) -> Tensor:
        """计算 N-gram 的哈希索引"""
        # ngram_embeds: [batch, seq_len, ngram_size, embed_dim]
        weights = self.hash_weights[head_idx]  # [ngram_size, embed_dim]

        # 加权求和后取模
        hash_input = (ngram_embeds * weights).sum(dim=(-2, -1))
        hash_idx = hash_input.abs().long() % self.embedding_tables[head_idx].size(0)
        return hash_idx

    def retrieve(self, input_embeds: Tensor) -> Tensor:
        """检索记忆向量"""
        batch, seq_len, embed_dim = input_embeds.shape

        # 提取 N-gram（滑动窗口）
        ngram_embeds = self._extract_ngrams(input_embeds)  # [B, L, N, D]

        # 多头哈希检索
        retrieved = []
        for h in range(self.num_heads):
            idx = self.compute_hash(ngram_embeds, h)  # [B, L]
            emb = self.embedding_tables[h][idx]       # [B, L, head_dim]
            retrieved.append(emb)

        # 拼接所有头
        return torch.cat(retrieved, dim=-1)  # [B, L, embed_dim]
```

**代码解读**：
1. **多头设计**：`num_heads=4` 意味着 4 个独立的哈希函数和嵌入表
2. **可学习哈希**：`hash_weights` 是可训练的，模型会学习最优的哈希映射
3. **大规模表**：`table_size=1e8` 支持 1 亿条记忆，但只激活 O(1) 条

### 2.3 上下文门控机制

查表得到的嵌入是上下文无关的静态信息。为处理语言的多义性（如"苹果"可指水果或公司），引入上下文感知的门控机制：

$$g_t = \sigma(W_g \cdot [h_t; e_t])$$
$$o_t = g_t \cdot W_v \cdot e_t$$

其中 h_t 为当前隐藏状态（已聚合全局上下文），g_t 为门控标量。当检索到的记忆与上下文不匹配时，门控值趋近于零，自动抑制噪声信息。

#### 💻 代码实现

**论文描述**：
> "We introduce context-aware gating to handle polysemy. The gate is computed from the concatenation of hidden state and retrieved memory."

**代码实现**:

```python
class ContextGating(nn.Module):
    """上下文感知的门控机制"""

    def __init__(self, embed_dim: int):
        super().__init__()
        # 门控网络：输入是 [hidden; memory] 的拼接
        self.gate_proj = nn.Linear(embed_dim * 2, 1)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden: Tensor, memory: Tensor) -> Tensor:
        """
        Args:
            hidden: 当前隐藏状态 [B, L, D]，已聚合全局上下文
            memory: 检索到的记忆 [B, L, D]，静态信息

        Returns:
            gated_output: 门控后的输出 [B, L, D]
        """
        # 计算门控值
        concat = torch.cat([hidden, memory], dim=-1)  # [B, L, 2D]
        gate = torch.sigmoid(self.gate_proj(concat))  # [B, L, 1]

        # 门控后的值
        value = self.value_proj(memory)  # [B, L, D]
        gated_output = gate * value      # [B, L, D]

        return gated_output
```

**代码解读**：
1. **输入拼接**：将 `hidden`（上下文）和 `memory`（静态记忆）拼接
2. **sigmoid 门控**：输出在 [0,1] 之间，0=忽略记忆，1=完全采纳
3. **值投影**：`value_proj` 将记忆投影到合适的表示空间

### 2.4 残差融合

门控后的记忆通过残差连接融合至主干网络：

$$h_t' = h_t + o_t$$

随后继续正常的 Attention 和 FFN/MoE 计算。

#### 💻 完整 Engram 层实现

```python
class EngramLayer(nn.Module):
    """完整的 Engram 层：Attention + Memory + MoE"""

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.memory = EngramMemory(config)
        self.gating = ContextGating(config.embed_dim)
        self.moe = MixtureOfExperts(config)

        self.attn_norm = RMSNorm(config.embed_dim)
        self.ffn_norm = RMSNorm(config.embed_dim)

    def forward(self, x: Tensor, input_embeds: Tensor) -> Tensor:
        # 1. 注意力层
        h = x + self.attention(self.attn_norm(x))

        # 2. 记忆检索 + 门控（核心创新）
        memory = self.memory.retrieve(input_embeds)
        memory_out = self.gating(h, memory)
        h = h + memory_out  # 残差连接

        # 3. MoE/FFN 层
        h = h + self.moe(self.ffn_norm(h))

        return h
```

**关键洞见**：
- 记忆检索使用 `input_embeds`（原始输入），而非 `h`（隐藏状态）
- 这使得索引是**确定性**的，支持预取优化

## 3. 稀疏分配策略

### 3.1 MoE 与 Engram 的参数分配

在固定参数预算下，MoE 专家与 Engram 记忆的分配比例直接影响模型性能。系统实验揭示了 **U 型曲线** 特征：

![稀疏分配实验](../style_comparison/images/a44c062d736c965077330a673248cfad4dfe5d671d50efcd868ddeb8dfec2342.jpg)

- **纯 MoE 配置**：缺乏专门的记忆模块，模型被迫用计算模拟查表
- **纯 Engram 配置**：失去条件计算能力，无法处理动态推理任务
- **最优配置**：约 75-80% 参数分配给 MoE，20-25% 分配给 Engram

#### 💻 训练配置

**27B 模型配置** (`configs/engram_27b.yaml`):

```yaml
model:
  embed_dim: 4096
  num_layers: 32
  num_heads: 32

  # MoE 配置
  moe:
    num_experts: 64
    top_k: 8

  # Engram 配置（核心）
  engram:
    enabled: true
    num_heads: 4
    table_size: 100000000  # 1 亿
    ngram_size: 3
    # 部署在哪些层
    layers: [4, 8, 12, 16, 20, 24, 28, 32]

training:
  batch_size: 2048
  learning_rate: 3e-4
  warmup_steps: 2000
  total_steps: 100000
```

**配置解读**：
- `layers: [4, 8, 12...]`：每 4 层插入一个 Engram 模块
- `table_size: 1e8`：1 亿条记忆，约 100B 参数（但激活参数很少）
- MoE 和 Engram 并行：75% 参数给 MoE，25% 给 Engram

### 3.2 无限内存扩展性

右图显示了固定计算量下仅扩展 Engram 容量的实验。验证损失呈现 log-linear 下降趋势，表明 Engram 具有良好的扩展性——增加内存可持续提升性能，且不增加推理计算量。

## 4. 实验结果与分析

### 4.1 主要基准测试

基于最优分配策略，训练 Engram-27B 与 MoE-27B 基线进行严格对比（等参数、等 FLOPs）：

| 任务类别 | 基准测试 | MoE-27B | Engram-27B | 提升 |
|---------|---------|---------|------------|------|
| 知识密集型 | MMLU | 71.2 | 74.6 | +3.4 |
| 知识密集型 | CMMLU | 68.9 | 72.9 | +4.0 |
| 通用推理 | BBH | 62.4 | 67.4 | +5.0 |
| 通用推理 | ARC-C | 84.1 | 87.8 | +3.7 |
| 代码生成 | HumanEval | 58.5 | 61.5 | +3.0 |
| 数学推理 | MATH | 35.2 | 37.6 | +2.4 |
| 长上下文 | NIAH (MQ) | 84.2 | 97.0 | +12.8 |

知识任务的提升符合预期。推理和代码任务的显著提升需要进一步分析。

### 4.2 推理任务受益机制：等效网络加深

LogitLens 和 CKA 分析揭示了推理任务受益的内在机制：

![LogitLens 和 CKA 分析](../style_comparison/images/3d7e70042b917eec0e3a9f6de58dd189cd1e655af876a05c5727ec5c932cad3f.jpg)

**LogitLens 分析**（左图）：测量各层隐状态与最终预测的 KL 散度。Engram 模型在早期层即达到低 KL 散度，表明特征组合过程显著加速。

**CKA 相似度分析**（中、右图）：Engram 第 5 层的表示相当于 MoE 第 12 层的表示。Engram 将静态模式识别从早期层"卸载"，**等效于增加 7 层网络深度**。

这解释了推理任务的提升：更多的网络层数可用于复杂推理，而非简单的模式识别。

### 4.3 功能分离验证

消融实验验证了 Engram 与 Transformer 主干的功能分离：

![Engram 消融实验](../style_comparison/images/df4aa413292b494b89fb94ab5d3a5d3d4af0e0801a2f2fc98a1433189fc41f27.jpg)

推理时完全关闭 Engram 模块：
- **事实知识任务**（TriviaQA、NaturalQuestions）：性能下降至 29-44%
- **阅读理解任务**（C3、RACE）：性能保持在 81-93%

结果表明 Engram 主要承担知识存储功能，而上下文依赖的任务主要依赖 Transformer 注意力机制，两者功能分离清晰。

### 4.4 门控激活模式可视化

门控机制的可视化分析进一步验证了 Engram 的工作模式：

![门控可视化](../style_comparison/images/72aafc116c62cdb1ea0b6fc393193ba8491b768e3ddb9a29cfe3ea9298c4b450.jpg)

高激活区域集中在：
- **命名实体**："Alexander the Great"、"the Milky Way"
- **固定搭配**："By the way"、"Princess of Wales"
- **中文成语与专名**："四大发明"、"张仲景"

这些静态模式被成功识别并通过查表获取，验证了架构设计的有效性。

## 5. 系统实现与效率优化

### 5.1 确定性索引的优势

与 MoE 的动态路由不同，Engram 的查表索引**仅取决于输入 token**，与隐藏状态无关。这一特性带来显著的系统优势：

![系统实现](../style_comparison/images/6408d0b8fea236ab4df4272a3366766b9f044cd1a5b589faf7a13e42d1f27df1.jpg)

**预取优化**：当前层计算时，可并行预取下一 Engram 层所需的嵌入向量。

**异构存储**：嵌入表可部署于 CPU 内存甚至 SSD，通过通信-计算重叠隐藏数据传输延迟。

#### 💻 预取机制实现

```python
class EngramMemoryWithPrefetch(EngramMemory):
    """支持预取的记忆模块"""

    def prefetch(self, input_ids: Tensor) -> None:
        """在当前层计算时，预取下一层需要的嵌入"""
        # 索引只依赖 input_ids，可以提前计算
        ngram_embeds = self.embed_ngrams(input_ids)

        for h in range(self.num_heads):
            idx = self.compute_hash(ngram_embeds, h)
            # 异步预取到 GPU 缓存
            self.embedding_tables[h].prefetch(idx)
```

**这就是论文中"确定性索引"的威力**：
- 索引只依赖输入 token，与隐藏状态无关
- 可以在当前层计算时预取下一层的嵌入
- 即使表在 CPU 内存，也能隐藏传输延迟

### 5.2 性能评估

将 100B 参数的嵌入表部署于 CPU 内存（通过 PCIe 传输），推理吞吐量仅下降 3%。结合自然语言的 Zipf 分布特性（少量高频模式占据大部分文本），可进一步采用分层缓存策略：高频嵌入置于 GPU HBM，低频嵌入置于 CPU 内存或 SSD。

## 6. 讨论与展望

### 6.1 架构设计的启示

Engram 的成功表明：**架构设计应匹配任务的内在结构**。语言建模本质上是异构的——静态知识适合查表，动态推理需要计算。统一处理不如分而治之。

### 6.2 与人脑机制的类比

Engram 架构与人脑的功能分区存在有趣的类比：海马体负责陈述性记忆，前额叶负责推理决策。Engram 可视为为 Transformer 引入的"人工海马体"。

### 6.3 未来研究方向

- **多模态扩展**：视觉模型是否存在类似的"静态模式 vs 动态推理"区分
- **知识编辑**：利用 Engram 的模块化特性实现高效的知识更新
- **稀疏性配比的自动学习**：根据任务特性自适应调整 MoE/Engram 比例

## 7. 实验复现

```bash
# 安装
git clone https://github.com/deepseek-ai/Engram
cd Engram
pip install -e .

# 训练（需要多卡）
torchrun --nproc_per_node=8 scripts/train.py \
    --config configs/engram_27b.yaml

# 推理
python scripts/inference.py \
    --model checkpoints/engram-27b \
    --prompt "中国的首都在"
```

## 8. 结论

本文提出 Engram 架构，通过引入条件记忆机制实现了静态知识与动态推理的功能分离。实验证明该架构在多个基准测试上显著优于同等规模的 MoE 模型，且具有良好的系统效率和扩展性。Engram 为大语言模型架构设计开辟了新的研究方向。

通过代码分析，我们可以看到 Engram 的核心设计：
1. **多头哈希**：可学习的哈希函数，降低冲突
2. **上下文门控**：sigmoid 门控处理多义性
3. **确定性索引**：支持预取，推理高效

代码实现与论文描述高度一致，工程质量较高。

---

**参考文献**

[1] DeepSeek-AI. Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models. 2025.

*代码开源地址：https://github.com/deepseek-ai/Engram*

---

*本文展示了 academic 风格 + 代码分析功能。在完整论文解读基础上，结合开源代码讲解实现细节，适合需要复现或深入理解实现的读者。*
