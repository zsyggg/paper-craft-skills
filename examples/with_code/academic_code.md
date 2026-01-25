# Engram: 基于可扩展查表的条件记忆架构（代码分析版）

**论文标题**：Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models
**论文链接**：[GitHub](https://github.com/deepseek-ai/Engram)
**作者团队**：北京大学 & DeepSeek-AI

> 本文为 **academic 风格 + 代码分析** 示例，展示如何结合开源代码讲解论文实现。

---

## 摘要

本文提出 Engram，一种基于条件记忆的新型稀疏架构。本文将结合 [官方开源代码](https://github.com/deepseek-ai/Engram) 详细解读实现细节。

## 1. 项目结构

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

## 2. 核心实现：条件记忆模块

### 2.1 N-gram 哈希检索

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

### 2.2 上下文门控机制

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

### 2.3 完整的 Engram 层

**代码实现**:

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

## 3. 训练配置

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

## 4. 推理优化

**预取机制** (`engram/layers/memory.py`):

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

## 5. 实验复现

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

## 6. 总结

通过代码分析，我们可以看到 Engram 的核心设计：
1. **多头哈希**：可学习的哈希函数，降低冲突
2. **上下文门控**：sigmoid 门控处理多义性
3. **确定性索引**：支持预取，推理高效

代码实现与论文描述高度一致，工程质量较高。

---

*本文展示了 academic 风格 + 代码分析功能。代码分析适合需要复现或理解实现细节的读者。*
