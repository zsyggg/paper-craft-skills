# Academic Paper Analyzer

<div align="center">

[English](README.md) | [中文](README_zh.md)

</div>

> Transform academic papers into deep technical articles with customizable styles
> High-precision parsing powered by MinerU Cloud API
> **Multiple writing styles, optional formula explanation, GitHub code analysis**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)

## Key Features

### Multiple Writing Styles
| Style | Description | Best For |
|-------|-------------|----------|
| **storytelling** | Intuitive, metaphors, like telling a story | Blogs, social media, general audience |
| **academic** | Professional terminology, rigorous | Academic reports, research sharing |
| **concise** | Direct, high information density | Quick overview, technical survey |

### Optional Enhancements
- **Formula Explanation**: Insert formula images with detailed symbol explanations
- **GitHub Code Analysis**: Clone repo, show key source code, code-paper alignment

### High-Precision Parsing
- **MinerU Cloud API**: Industry-leading PDF parsing accuracy
- **Complete Extraction**: Images, tables, and LaTeX formulas
- **Smart Recognition**: Automatically extract paper structure and metadata

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install requests markdown
```

2. Get your MinerU API Token:
   - Visit [MinerU Official Site](https://mineru.net) to register
   - Set environment variable:
     ```bash
     export MINERU_TOKEN="your_token_here"
     ```

### Usage as Claude Code Skill

Simply tell Claude:

```
Please analyze this paper: /path/to/paper.pdf
```

Claude will ask you to choose:
1. **Writing style**: storytelling / academic / concise
2. **Formula explanation**: yes / no
3. **GitHub code analysis**: yes / no (if repo available)

Then automatically generate the article.

## Example

See [examples/Engram_Analysis](examples/Engram_Analysis) for a complete example.

## Writing Styles

### storytelling (Default)

Start from intuition, use metaphors, like telling a story:

```markdown
Have you ever thought that large language models don't actually have memory?
Every time they answer "What's the capital of China?", they have to "compute"
the answer from scratch. Isn't that using a cannon to kill a mosquito?
```

### academic

Professional terminology, rigorous expression:

```markdown
This paper proposes Engram, a conditional memory architecture based on
scalable lookup. The method combines N-gram retrieval with Transformer
backbone, achieving O(1) complexity for static knowledge acquisition.
```

### concise

Direct, high information density:

```markdown
**Core Innovation**: O(1) lookup mechanism in Transformer
**Key Design**: N-gram hash retrieval + context gating
**Results**: MMLU +3.4, BBH +5.0 (equal params)
```

## Output Formats

| Format | Description |
|--------|-------------|
| **Markdown** (default) | Lightweight, easy to edit |
| **HTML** (optional) | Embedded images, single-file sharing |

## Architecture

```
PDF → MinerU API → Markdown + Images
                        ↓
              Style Selection (user)
                        ↓
              Article Generation
                        ↓
              Markdown / HTML Output
```

## Scripts

| Script | Purpose |
|--------|---------|
| `mineru_api.py` | MinerU Cloud API (recommended) |
| `extract_paper_info.py` | Extract paper metadata |
| `generate_html.py` | Markdown → HTML |

## API Limits

- Max file size: 200MB
- Max pages: 600
- Supports: PDF, DOC, PPT, images

## Links

- [MinerU](https://mineru.net)
- [Example](examples/Engram_Analysis)

## License

MIT
