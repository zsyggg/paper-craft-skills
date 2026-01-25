# paper-craft-skills

English | [中文](./README.zh.md)

Claude Code skills for academic papers: deep analysis, comics, summaries and more.

**Compatible with**: Cursor, Codex, Windsurf, and other AI coding assistants that support Claude Code skills.

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
| [paper-comic](#paper-comic) | Generate educational comics from papers | ✅ Available |

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

Generate educational comics from academic papers with visual storytelling.

### Features

| Feature | Description |
|---------|-------------|
| **4 Art Styles** | classic / tech / warm / chalk |
| **10-Page Comics** | Complete narrative with characters |
| **Visual Metaphors** | Abstract concepts made tangible |
| **Chinese Text** | All dialogue in Chinese |

### Art Styles

| Style | Best For | Example |
|-------|----------|---------|
| **classic** | General papers | ![classic](examples/paper-comic/cover-classic.png) |
| **tech** | AI/CS papers | ![tech](examples/paper-comic/cover-tech.png) |
| **warm** | Psychology/Education | ![warm](examples/paper-comic/cover-warm.png) |
| **chalk** | Math/Physics | ![chalk](examples/paper-comic/cover-chalk.png) |

### Usage

```
/paper-comic /path/to/paper.pdf
/paper-comic /path/to/paper.pdf --style tech
```

### Example: Engram Paper (10 pages)

| Page | Content |
|------|---------|
| ![Cover](examples/paper-comic/00-cover.png) | Cover |
| ![Page 1](examples/paper-comic/01-page.png) | Problem introduction |
| ![Page 5](examples/paper-comic/05-page.png) | Core concept |
| ![Page 10](examples/paper-comic/10-page.png) | Conclusion |

---

## License

MIT
