# 📚 Academic Paper Analyzer - 5-Minute Paper Reading Tool

<div align="center">

[English](README.md) | [中文](README_zh.md)

</div>

> High-precision paper parsing powered by MinerU Cloud API
> Transform academic papers into concise, fluent technical articles
> **Perfect for academic presentations, paper sharing, and quick research understanding**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)

## ✨ Key Features

### 🎯 5-Minute Reading Experience
- **Concise Content**: Distill lengthy papers into 5-minute readable technical articles
- **Natural Narrative**: Replace rigid academic language with storytelling approach
- **Intuition First**: Start from problems and intuition, not technical jargon

### 🔬 High-Precision Parsing
- **MinerU Cloud API**: Industry-leading PDF parsing accuracy
- **Complete Extraction**: Images, tables, and LaTeX formulas - nothing missed
- **Smart Recognition**: Automatically extract paper structure and metadata

### 📝 Professional Writing Paradigm
- **Share-Ready**: Perfect for academic presentations, blog posts, and technical sharing
- **Logical Progression**: From simple to complex, layer by layer
- **Vivid Metaphors**: Make abstract concepts concrete and understandable

## 🚀 Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install requests markdown
```

2. Get your MinerU API Token:
   - Visit [MinerU Official Site](https://mineru.net) to register
   - Obtain your API Token
   - Set environment variable:
     ```bash
     export MINERU_TOKEN="your_token_here"
     ```

### Usage as Claude Code Skill

This tool is designed as a **Claude Code Skill**. Simply tell Claude:

```
Please analyze this paper: /path/to/paper.pdf
```

Or:

```
Help me create a 5-minute reading article for this academic paper
```

Claude will automatically:
1. ✅ Parse the PDF using MinerU API
2. ✅ Extract paper structure, figures, and formulas
3. ✅ Generate a natural, fluent article following the writing paradigm
4. ✅ Output in Markdown format (HTML optional)

**No manual commands needed!** The skill handles everything for you.

## 📖 Example Case

Check out the [Engram Paper Analysis Example](examples/Engram_Analysis) to see the complete analysis effect.

This example demonstrates how to transform a complex deep learning paper into:
- ⚡ 30-second overview
- 🎯 Core problem and solution
- 📊 Key experimental results
- 💡 In-depth technical insights

## 🎨 Writing Paradigm

### ❌ Avoid Writing Like This

```markdown
## Method

This paper proposes a hash-table-based conditional memory module.

Main innovations include:
1. Scalable memory bank design
2. Efficient lookup mechanism
3. Dynamic memory update strategy

In experiments, we validated the effectiveness across multiple datasets...
```

### ✅ Write Like This Instead

```markdown
## Why Do We Need This Innovation?

Have you ever thought that large language models don't actually have memory?
Every time they answer "What's the capital of China?", they have to "compute"
the answer from scratch, rather than looking it up like a dictionary.

It's like deriving word meanings from etymology every time, instead of just
checking a dictionary. Isn't that using a cannon to kill a mosquito?

## Core Innovation: Memory is Memory, Computation is Computation

Engram's core idea is simple: put knowledge that can be "memorized" into
a lookup table...
```

### Key Principles

1. **Start with Intuition**: Present problems and motivations before technical details
2. **Explain Background**: Introduce old technologies before new ones
3. **Use Simple Examples**: Choose one example and use it throughout
4. **Vivid Metaphors**: Make abstract concepts concrete ("cannon to kill mosquito", "lookup vs. derive")
5. **Logical Progression**: Layer by layer, let readers naturally understand the design
6. **Distill Insights**: Summarize core ideas in one sentence ("Memory is memory, computation is computation")
7. **Rich Illustrations**: Explain every key figure in detail

## 📁 Output Formats

### Markdown (Default)
- Lightweight, easy to edit
- Compatible with WeChat Official Account
- Perfect for daily use

### HTML (Optional)
- Embedded images (base64)
- Single-file sharing
- Great for preview and sharing

## 🛠️ Technical Architecture

```
┌─────────────────────────────────────────────┐
│  Input: PDF Paper                            │
├─────────────────────────────────────────────┤
│  Step 1: MinerU API High-Precision Parsing   │
│  - Extract text, images, tables, formulas    │
│  - Preserve LaTeX formulas                   │
├─────────────────────────────────────────────┤
│  Step 2: Smart Metadata Extraction           │
│  - Title, authors, institutions              │
│  - Section structure                         │
│  - Figure list                               │
├─────────────────────────────────────────────┤
│  Step 3: Claude Intelligent Writing          │
│  - Analyze paper content                     │
│  - Generate article following paradigm       │
│  - Natural and fluent narrative style        │
├─────────────────────────────────────────────┤
│  Step 4: Multi-Format Output                 │
│  - Markdown (default)                        │
│  - HTML (optional, base64 embedded images)   │
└─────────────────────────────────────────────┘
```

## 📊 Article Structure Template

```markdown
# [Paper Title] - In-Depth Analysis

> **One-Sentence Summary**: [Core innovation]
> **Code**: [GitHub link]
> **Paper**: [Conference/Journal] [Year]

## ⚡ 30-Second Overview

**Problem**: [Limitations of existing approaches]
**Solution**: [Core innovation of the paper]
**Key Results**: [Most impressive experimental results]

## 🎯 Why Does This Matter?

[Start from intuition and problems to engage readers]

## 📖 Background

[Explain foundational technologies with simple examples]

## 💡 Core Innovation

[Detail innovations with illustrations]

## 📊 Experimental Validation

[Key experimental results and data analysis]

## 🔬 In-Depth Analysis

[Mechanism analysis, ablation studies, etc.]

## 💭 Thoughts and Future Directions

[Distill core insights, personal perspectives]
```

## 🎓 Best Practices

### 1. Writing Style
- ✅ Natural paragraph narration, like storytelling
- ✅ Start from intuition and problems
- ✅ Vivid metaphors and simple examples
- ❌ Avoid bullet lists ("1. 2. 3.")
- ❌ Avoid AI clichés ("delve into", "crucial", "in the domain of")
- ❌ Avoid mechanical section titles

### 2. Image Usage
- ✅ Use extracted images for formulas, not LaTeX syntax
- ✅ Use screenshots for result tables, more accurate than reformatting
- ✅ Explain every key figure
- ❌ Don't just paste 1-2 framework diagrams

### 3. Context Saving
- ✅ Trust metadata in `paper_info.json`
- ✅ Reference image paths directly, don't read repeatedly
- ✅ Only examine 1-2 core architecture diagrams
- ❌ Don't repeatedly read all image files

## 📚 Scripts Reference

| Script | Function | Purpose |
|--------|----------|---------|
| `mineru_api.py` | MinerU Cloud API call | High-precision PDF parsing (recommended) |
| `convert_pdf.py` | Local PDF conversion | Alternative, requires PyMuPDF |
| `extract_paper_info.py` | Extract paper metadata | Generate structured paper info |
| `generate_html.py` | Markdown → HTML | Generate HTML with base64 embedded images |

*Note: When using as a Claude Code Skill, these scripts are called automatically by Claude.*

## ⚠️ Notes

### API Limitations
- Maximum file size: 200MB
- Maximum pages: 600
- Supports PDF, DOC, PPT, images, etc.

### Usage Tips
- Prefer MinerU API for highest accuracy
- The skill automatically handles the entire workflow
- Generated articles follow the natural storytelling paradigm
- Select 3-5 key figures for illustration

## 🔗 Related Links

- [MinerU Official Site](https://mineru.net)
- [MinerU GitHub](https://github.com/opendatalab/MinerU)
- [Example: Engram Paper Analysis](examples/Engram_Analysis)

## 📄 License

MIT License

---

**Use Cases**
- 📢 Academic presentation preparation
- 📱 Technical sharing on social media
- 📝 Technical blog writing
- 🎓 Quick paper understanding
- 💡 Research direction exploration

**Keywords**: Paper Analysis, PDF Parsing, Academic Sharing, Technical Blog, 5-Minute Reading
