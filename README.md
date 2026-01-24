# Academic Paper Analyzer

<div align="center">

[English](README.md) | [中文](README_zh.md)

![Hero Banner](images/hero_banner.png)

**Transform academic papers into deep technical articles with customizable styles**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://python.org)

</div>

---

## Features

### Three Writing Styles

![Writing Styles Comparison](images/styles_comparison.png)

Choose the style that best fits your audience:

| Style | Description | Best For |
|-------|-------------|----------|
| **storytelling** | Start from intuition, use metaphors, like telling a story to a friend | Tech blogs, WeChat articles, general audience |
| **academic** | Professional terminology, rigorous structure, preserves original concepts | Research reports, paper reviews, academic sharing |
| **concise** | Direct to the point, tables and lists, high information density | Quick overview, technical survey, busy readers |

<details>
<summary><b>See style examples</b></summary>

**Storytelling style:**
> Have you ever thought about this: when we ask ChatGPT "What's the capital of China?", how does it know the answer? Intuitively, we might think the model "remembers" this knowledge. But actually, LLMs don't have a "memory module" — all answers are **computed**...

**Academic style:**
> This paper proposes Engram, a conditional memory architecture based on scalable lookup. The method integrates N-gram retrieval with Transformer backbone, achieving O(1) complexity for static knowledge acquisition while preserving dynamic reasoning capabilities...

**Concise style:**
> | Component | Function | Complexity |
> |-----------|----------|------------|
> | N-gram retrieval | Hash mapping to embedding table | O(1) |
> | Context gating | Hidden state determines adoption | O(d) |
> | Residual fusion | Add to backbone network | O(d) |

[View complete examples →](examples/style_comparison)

</details>

### Formula Explanation (Optional)

![Formula Feature](images/formula_feature.png)

When enabled, the tool will:
- Insert formula images extracted from the paper
- Provide detailed symbol explanations
- Connect formulas to intuitive understanding

<details>
<summary><b>Example: Formula explanation in action</b></summary>

```markdown
The gating mechanism is computed as:

![Gating Formula](images/formula_gating.png)

**Symbol breakdown:**
- g_t: gating scalar (0-1), controls how much memory to use
- h_t: current hidden state (already aggregated global context)
- e_t: retrieved embedding from memory table
- σ: sigmoid function, squashes output to [0,1]

**Intuition:** When the retrieved memory conflicts with current context
(e.g., discussing tech companies but retrieved "apple" as fruit),
the gate closes (g_t → 0), suppressing the noise.
```

</details>

### GitHub Code Analysis (Optional)

![Code Analysis Feature](images/code_feature.png)

When the paper has an open-source implementation:
- Clone the repository automatically
- Extract key source code snippets
- Align code with paper concepts
- Show implementation details

<details>
<summary><b>Example: Code-paper alignment</b></summary>

```markdown
## Implementation: N-gram Hash Retrieval

The paper describes the hash function as "multi-head hashing to reduce collision".
Here's the actual implementation:

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
        # Multi-head hashing: each head uses different hash function
        embeddings = []
        for h in range(self.num_heads):
            idx = self.hash_fn(ngram_ids, self.hash_weights[h])
            embeddings.append(self.embedding_tables[h][idx])
        return torch.cat(embeddings, dim=-1)
​```

**Key insight:** The multi-head design ensures that even if one hash function
produces a collision, others likely won't, improving overall retrieval quality.
```

</details>

### High-Precision PDF Parsing

Powered by **MinerU Cloud API**:
- Industry-leading PDF parsing accuracy
- Complete extraction of images, tables, and LaTeX formulas
- Smart recognition of paper structure and metadata
- Supports PDF, DOC, PPT, and images (max 200MB, 600 pages)

## Quick Start

### Prerequisites

```bash
pip install requests markdown
```

### Get MinerU API Token

1. Visit [MinerU Official Site](https://mineru.net) to register
2. Get your API token from dashboard
3. Set environment variable:
   ```bash
   export MINERU_TOKEN="your_token_here"
   ```

### Usage

Simply tell Claude Code:

```
Please analyze this paper: /path/to/paper.pdf
```

Claude will guide you through:

1. **Style selection**: storytelling / academic / concise
2. **Formula explanation**: yes / no
3. **Code analysis**: yes / no (if GitHub repo detected)

Then automatically generate the article with extracted images.

## Examples

### Complete Analysis

- [Engram Analysis (storytelling)](examples/Engram_Analysis) - Full paper breakdown with images

### Style Comparison

Same paper (Engram) in three different styles:

| Style | Lines | Images | Link |
|-------|-------|--------|------|
| [storytelling](examples/style_comparison/storytelling.md) | ~200 | 6 | Narrative, metaphors |
| [academic](examples/style_comparison/academic.md) | ~180 | 6 | Formal, structured |
| [concise](examples/style_comparison/concise.md) | ~110 | 6 | Tables, bullet points |

## Output Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| **Markdown** (default) | Lightweight, easy to edit | WeChat articles, blogs |
| **HTML** (optional) | Base64 embedded images, single file | Preview, sharing |

Generate HTML after Markdown:
```bash
python scripts/generate_html.py article.md article.html
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  PDF Paper  │────▶│ MinerU API  │────▶│   Markdown  │
└─────────────┘     └─────────────┘     │  + Images   │
                                        └──────┬──────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    ▼                          ▼                          ▼
            ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
            │  storytelling │          │   academic    │          │    concise    │
            │    Style      │          │    Style      │          │    Style      │
            └───────────────┘          └───────────────┘          └───────────────┘
                    │                          │                          │
                    └──────────────────────────┼──────────────────────────┘
                                               ▼
                                    ┌─────────────────────┐
                                    │  Article + Images   │
                                    │  (MD / HTML)        │
                                    └─────────────────────┘
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/mineru_api.py` | MinerU Cloud API for PDF parsing |
| `scripts/extract_paper_info.py` | Extract paper metadata (title, authors, etc.) |
| `scripts/generate_html.py` | Convert Markdown to HTML with embedded images |

## Links

- [MinerU](https://mineru.net) - PDF parsing API
- [Examples](examples/) - Sample outputs
- [Style Guides](styles/) - Writing style definitions

## License

MIT
