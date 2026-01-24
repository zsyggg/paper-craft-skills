# Academic Paper Analyzer - 学术论文深度解析

<div align="center">

[English](README.md) | [中文](README_zh.md)

![Hero Banner](images/hero_banner.png)

**将学术论文转化为深度技术文章，支持多种写作风格**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)

</div>

---

## 核心功能

### 三种写作风格

![风格对比](images/styles_comparison.png)

根据目标读者选择最合适的风格：

| 风格 | 特点 | 适用场景 |
|------|------|----------|
| **storytelling** | 从直觉出发，用比喻讲故事，像跟朋友聊天 | 技术博客、微信公众号、科普文章 |
| **academic** | 专业术语，严谨结构，保留原文概念 | 研究报告、论文综述、学术分享 |
| **concise** | 直击核心，表格列表，信息密度高 | 快速了解、技术调研、忙碌读者 |

<details>
<summary><b>查看风格示例</b></summary>

**storytelling 风格：**
> 你有没有想过这个问题：当我们问 ChatGPT "中国的首都在哪里"，它是怎么知道答案的？直觉上，我们可能会觉得模型"记住"了这个知识。但实际上，大模型根本没有这样的"记忆模块"——它所有的答案都是**算**出来的...

**academic 风格：**
> 本文提出 Engram，一种基于条件记忆（Conditional Memory）的新型稀疏架构。该方法将 N-gram 查表机制与 Transformer 主干网络相结合，通过可扩展的哈希检索实现 O(1) 复杂度的静态知识获取，同时保留 Transformer 的动态推理能力...

**concise 风格：**
> | 步骤 | 操作 | 复杂度 |
> |------|------|--------|
> | 1. N-gram 检索 | 哈希映射到嵌入表 | O(1) |
> | 2. 上下文门控 | 隐藏状态决定采纳度 | O(d) |
> | 3. 残差融合 | 加到主干网络 | O(d) |

[查看完整示例 →](examples/style_comparison)

</details>

### 公式讲解（可选）

![公式讲解功能](images/formula_feature.png)

启用后，工具会：
- 插入论文中提取的公式图片
- 详细解读每个符号的含义
- 将公式与直观理解联系起来

<details>
<summary><b>示例：公式讲解效果</b></summary>

```markdown
门控机制的计算公式为：

![门控公式](images/formula_gating.png)

**符号解读：**
- g_t: 门控标量 (0-1)，控制记忆使用量
- h_t: 当前隐藏状态（已聚合全局上下文）
- e_t: 从记忆表检索到的嵌入向量
- σ: sigmoid 函数，将输出压缩到 [0,1]

**直观理解：** 当检索到的记忆与当前上下文矛盾时
（比如在讨论科技公司时检索到"苹果"是水果），
门控值趋近于零（g_t → 0），自动抑制噪声。
```

</details>

### GitHub 代码分析（可选）

![代码分析功能](images/code_feature.png)

当论文有开源实现时：
- 自动克隆仓库
- 提取关键源码片段
- 代码与论文概念对照
- 展示实现细节

<details>
<summary><b>示例：代码-论文对照</b></summary>

```markdown
## 实现细节：N-gram 哈希检索

论文描述哈希函数为"多头哈希以减少冲突"。
以下是实际实现：

​```python
# From engram/layers/memory.py
class EngramMemory(nn.Module):
    def __init__(self, num_heads=4, embed_dim=4096, table_size=100_000_000):
        self.hash_weights = nn.Parameter(torch.randn(num_heads, 3, embed_dim))
        self.embedding_tables = nn.ParameterList([
            nn.Parameter(torch.randn(table_size, embed_dim // num_heads))
            for _ in range(num_heads)
        ])

    def forward(self, ngram_ids):
        # 多头哈希：每个头使用不同的哈希函数
        embeddings = []
        for h in range(self.num_heads):
            idx = self.hash_fn(ngram_ids, self.hash_weights[h])
            embeddings.append(self.embedding_tables[h][idx])
        return torch.cat(embeddings, dim=-1)
​```

**关键洞见：** 多头设计确保即使某个哈希函数产生冲突，
其他头很可能不会，从而提高整体检索质量。
```

</details>

### 高精度 PDF 解析

基于 **MinerU Cloud API**：
- 业界领先的 PDF 解析精度
- 完整提取图片、表格、LaTeX 公式
- 智能识别论文结构和元数据
- 支持 PDF、DOC、PPT、图片（最大 200MB，600 页）

## 快速开始

### 前置准备

```bash
pip install requests markdown
```

### 获取 MinerU API Token

1. 访问 [MinerU 官网](https://mineru.net) 注册
2. 从控制台获取 API Token
3. 设置环境变量：
   ```bash
   export MINERU_TOKEN="your_token_here"
   ```

### 使用方式

告诉 Claude Code：

```
请帮我分析这篇论文：/path/to/paper.pdf
```

Claude 会引导你完成：

1. **风格选择**：storytelling / academic / concise
2. **公式讲解**：是 / 否
3. **代码分析**：是 / 否（如检测到 GitHub 仓库）

然后自动生成带图片的文章。

## 示例

### 完整分析

- [Engram 分析 (storytelling)](examples/Engram_Analysis) - 完整论文深度解读

### 风格对比

同一篇论文（Engram）的三种不同风格：

| 风格 | 行数 | 图片数 | 链接 |
|------|------|--------|------|
| [storytelling](examples/style_comparison/storytelling.md) | ~200 | 6 | 故事叙述，生动比喻 |
| [academic](examples/style_comparison/academic.md) | ~180 | 6 | 正式严谨，结构清晰 |
| [concise](examples/style_comparison/concise.md) | ~110 | 6 | 表格列表，信息密集 |

## 输出格式

| 格式 | 说明 | 使用场景 |
|------|------|----------|
| **Markdown**（默认） | 轻量、易编辑 | 微信公众号、博客 |
| **HTML**（可选） | Base64 嵌入图片，单文件 | 预览、分享 |

生成 HTML：
```bash
python scripts/generate_html.py article.md article.html
```

## 架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PDF 论文   │────▶│ MinerU API  │────▶│  Markdown   │
└─────────────┘     └─────────────┘     │  + 图片     │
                                        └──────┬──────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
            ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
            │  storytelling │          │   academic    │          │    concise    │
            │    故事风格   │          │    学术风格   │          │    精炼风格   │
            └───────────────┘          └───────────────┘          └───────────────┘
                    │                          │                          │
                    └──────────────────────────┼──────────────────────────┘
                                               ▼
                                    ┌─────────────────────┐
                                    │   文章 + 图片       │
                                    │   (MD / HTML)       │
                                    └─────────────────────┘
```

## 脚本

| 脚本 | 用途 |
|------|------|
| `scripts/mineru_api.py` | MinerU Cloud API 调用 |
| `scripts/extract_paper_info.py` | 提取论文元数据（标题、作者等） |
| `scripts/generate_html.py` | Markdown 转 HTML（嵌入图片） |

## 链接

- [MinerU](https://mineru.net) - PDF 解析 API
- [示例](examples/) - 输出样例
- [风格指南](styles/) - 写作风格定义

## License

MIT
