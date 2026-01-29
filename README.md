# paper-craft-skills

English | [中文](./README.zh.md)

**Turn academic papers into in-depth articles or educational comics with one click**

<table>
<tr>
<td width="50%" align="center">
<b>paper-analyzer</b><br/>
Paper → Deep Article<br/><br/>
<img src="images/hero_banner.png" width="400"/>
</td>
<td width="50%" align="center">
<b>paper-comic</b><br/>
Paper → 10-Page Comic<br/><br/>
<img src="examples/paper-comic/00-cover.png" width="200"/>
</td>
</tr>
</table>

## Quick Start

### 1. Install

```bash
npx skills add zsyggg/paper-craft-skills
```

### 2. Configure API

| Skill | Required |
|-------|----------|
| paper-analyzer | `MINERU_TOKEN` - Get from [mineru.net](https://mineru.net) |
| paper-comic | Google Account (auto login on first run) |

```bash
# For paper-analyzer
pip install requests markdown
export MINERU_TOKEN="your_token"
```

### 3. Use

```bash
# Analyze paper, generate article
Please analyze this paper: /path/to/paper.pdf

# Generate paper comic
Please turn this paper into a comic: /path/to/paper.pdf
```

---

## paper-analyzer

Transform academic papers into deep technical articles with 3 writing styles.

### Styles

<table>
<tr>
<td align="center" width="33%">
<b>academic</b><br/>(default)<br/><br/>
Formal, rigorous
</td>
<td align="center" width="33%">
<b>storytelling</b><br/><br/><br/>
Vivid metaphors
</td>
<td align="center" width="33%">
<b>concise</b><br/><br/><br/>
Tables, bullet points
</td>
</tr>
</table>

![Style Comparison](images/styles_comparison.png)

### Optional Features

| Feature | Description |
|---------|-------------|
| **Formula Explanation** | Insert formula images with symbol breakdown |
| **Code Analysis** | Align paper concepts with GitHub source code |

---

## paper-comic

Transform academic papers into 10-page educational comics with 4 art styles.

### Art Styles

<table>
<tr>
<td align="center" width="25%"><img src="examples/paper-comic/cover-classic.png" width="150"/><br/><b>classic</b><br/>General (default)</td>
<td align="center" width="25%"><img src="examples/paper-comic/cover-tech.png" width="150"/><br/><b>tech</b><br/>AI/CS</td>
<td align="center" width="25%"><img src="examples/paper-comic/cover-warm.png" width="150"/><br/><b>warm</b><br/>Psychology</td>
<td align="center" width="25%"><img src="examples/paper-comic/cover-chalk.png" width="150"/><br/><b>chalk</b><br/>Math/Physics</td>
</tr>
</table>

### Example: Engram Paper

<table>
<tr>
<td align="center"><img src="examples/paper-comic/00-cover.png" width="100"/><br/>Cover</td>
<td align="center"><img src="examples/paper-comic/01-page.png" width="100"/><br/>1</td>
<td align="center"><img src="examples/paper-comic/02-page.png" width="100"/><br/>2</td>
<td align="center"><img src="examples/paper-comic/03-page.png" width="100"/><br/>3</td>
<td align="center"><img src="examples/paper-comic/04-page.png" width="100"/><br/>4</td>
<td align="center"><img src="examples/paper-comic/05-page.png" width="100"/><br/>5</td>
</tr>
<tr>
<td align="center"><img src="examples/paper-comic/06-page.png" width="100"/><br/>6</td>
<td align="center"><img src="examples/paper-comic/07-page.png" width="100"/><br/>7</td>
<td align="center"><img src="examples/paper-comic/08-page.png" width="100"/><br/>8</td>
<td align="center"><img src="examples/paper-comic/09-page.png" width="100"/><br/>9</td>
<td align="center"><img src="examples/paper-comic/10-page.png" width="100"/><br/>10</td>
<td></td>
</tr>
</table>

---

## Compatibility

Works with:

- Claude Code
- Cursor
- Codex
- Windsurf
- Other Claude Code skill-compatible tools

---

## Acknowledgments

- **baoyu-gemini-web** - Image generation backend, based on [JimLiu/baoyu-skills](https://github.com/JimLiu/baoyu-skills)
- **MinerU** - High-precision PDF parsing from [mineru.net](https://mineru.net)

---

## License

MIT