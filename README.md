# paper-craft-skills

English | [中文](./README.zh.md)

Claude Code skills for academic papers: deep analysis, comics, summaries and more.

## Installation

### Quick Install (Recommended)

```bash
npx skills add zsyggg/paper-craft-skills
```

### Manual Install

Simply tell Claude Code:

> Please install skills from github.com/zsyggg/paper-craft-skills

## Available Skills

| Skill | Description | Status |
|-------|-------------|--------|
| [paper-analyzer](#paper-analyzer) | Transform papers into readable articles with multiple styles | ✅ Available |
| [paper-comic](#paper-comic) | Generate educational comics from papers | 🚧 Coming Soon |

---

## paper-analyzer

Transform academic papers into deep technical articles with customizable styles.

![Hero Banner](images/hero_banner.png)

### Features

| Feature | Description |
|---------|-------------|
| **3 Writing Styles** | storytelling (narrative) / academic (formal) / concise (dense) |
| **Formula Explanation** | Insert formula images with symbol breakdown |
| **Code Analysis** | Align paper concepts with GitHub source code |
| **High-Precision Parsing** | MinerU Cloud API for PDF/images/tables/LaTeX |

### Usage

```
Please analyze this paper: /path/to/paper.pdf
```

Claude will ask you to choose:
1. **Style**: academic (default) / storytelling / concise
2. **Formula explanation**: yes / no
3. **Code analysis**: yes / no (if GitHub repo detected)

### Style Comparison

Same paper in three different styles:

![Styles Comparison](images/styles_comparison.png)

| Style | Description | Example |
|-------|-------------|---------|
| **academic** | Formal, structured (default) | [View](examples/style_comparison/academic.md) |
| storytelling | Narrative, metaphors | [View](examples/style_comparison/storytelling.md) |
| concise | Tables, bullet points | [View](examples/style_comparison/concise.md) |

### Optional Features

#### Formula Explanation

Insert formula images with detailed symbol breakdown.

![Formula Feature](images/formula_feature.png)

→ [View example: academic + formulas](examples/with_formulas/academic_formulas.md)

#### Code Analysis

Align paper concepts with GitHub source code.

![Code Feature](images/code_feature.png)

→ [View example: academic + code](examples/with_code/academic_code.md)

### Prerequisites

```bash
pip install requests markdown
export MINERU_TOKEN="your_token_here"  # Get from https://mineru.net
```

---

## paper-comic

🚧 **Coming Soon**

Generate educational comics from academic papers, explaining innovations and background in visual storytelling format.

Planned features:
- Multiple comic styles (Logicomix, manga guide, etc.)
- Panel-by-panel breakdown of paper concepts
- Character-driven explanations

---

## License

MIT
