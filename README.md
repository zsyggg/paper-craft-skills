# paper-craft-skills

English | [中文](./README.zh.md)

**Turn academic papers into polished method figures and in-depth articles — zero config, one command.**

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="900" alt="Transformer architecture overview — generated from Attention Is All You Need"/>
</p>

<p align="center">
  <b>From arxiv link to publication-ready visuals and deep-dive articles.</b><br/>
  Drop a paper URL, pick a style, get figures and articles that look like a human expert made them.
</p>

---

## What's inside

<table>
<tr>
<td width="50%" align="center" valign="top">
<img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="400"/><br/>
<h3>paper-comic</h3>
Paper → Method figures & visual notes<br/>
<sub>sketchnote · paper-figure styles</sub>
</td>
<td width="50%" align="center" valign="top">
<h3>paper-analyzer</h3>
Paper → Deep technical articles<br/>
<sub>storytelling · academic · concise styles</sub><br/><br/>
<img src="images/styles_comparison.png" width="400"/>
</td>
</tr>
</table>

---

## paper-comic — Method figures that explain

**Not a comic book. Not a diagram tool. It reads your paper, figures out what's worth drawing, and draws it.**

### How it works

```text
You: /paper-comic https://arxiv.org/abs/1706.03762

Paper-comic reads the paper, then recommends:

  I suggest 3 figures:
  1. Transformer architecture overview
  2. Self-attention mechanism (the core innovation)
  3. Multi-head attention detail

  Language? [Chinese / English]  Style? [sketchnote / paper-figure]  Generate all 3?

You confirm. It draws.
```

### Two styles

| Style | What it looks like | When to use |
|-------|-------------------|-------------|
| **paper-figure** | Clean, publication-grade diagrams — polished modules, precise arrows, conference-paper quality | README hero images, slides, tweets, arXiv pages |
| **sketchnote** | Warm hand-drawn research notes — personal, approachable, feels like a colleague's whiteboard sketch | Blog posts, explainer threads, teaching, quick understanding |

### Real example: Attention Is All You Need

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="700"/>
</p>

<p align="center"><b>paper-figure style — Transformer architecture overview</b></p>

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/self-attention-sketchnote.png" width="450"/>
</p>

<p align="center"><b>sketchnote style — Self-attention mechanism explained</b></p>

> Full example: [examples/paper-illustrated/attention-is-all-you-need](./examples/paper-illustrated/attention-is-all-you-need)

---

## paper-analyzer — Articles that read like a human expert wrote them

Transform any paper into a polished, in-depth article. **Not a paper translator — a re-interpreter.** It reads the full text, searches for open-source code, cross-references implementations, and writes with your chosen style.

### Three writing styles

| Style | Output | Best for |
|-------|--------|----------|
| **storytelling** | Blog-style with hooks, analogies, and a golden takeaway sentence | WeChat articles, Twitter threads, blog posts |
| **academic** | Structured deep-dive with KaTeX formulas, comparison tables, and critical analysis | Lab meetings, literature reviews, research notes |
| **concise** | Dense summary with a Mermaid architecture diagram and key data table | Quick understanding, pre-reading skim |

### What it actually does

- 📖 Reads the full paper (arxiv URL, local PDF, or pasted text)
- 🔍 Searches GitHub for the paper's open-source implementation
- ✍️ Writes with vivid analogies (storytelling) or rigorous depth (academic)
- 📐 Renders formulas with KaTeX — clean, readable, professional
- 📊 Adds Mermaid diagrams to explain architectures at a glance
- 🌐 Outputs **HTML** — ready to share, read on mobile, or post anywhere
- ⚡ **Zero config** — no API keys required

### Input formats

```bash
/paper-analyzer https://arxiv.org/abs/1706.03762     # arxiv link
/paper-analyzer /path/to/paper.pdf                     # local PDF
/paper-analyzer                                         # then paste text
```

---

## Quick Start

```bash
# One command to install
npx skills add zsyggg/paper-craft-skills

# One command to use
/paper-comic https://arxiv.org/abs/1706.03762
/paper-analyzer https://arxiv.org/abs/1706.03762
```

**That's it. No API keys. No accounts. No setup.** Image generation uses whatever your environment already has — Codex built-in `imagegen`, or any installed image backend.

---

## Compatibility

- **Codex** (recommended — built-in image generation for paper-comic)
- **Claude Code**
- **Cursor**
- **Windsurf**
- Any Claude Code skill-compatible tool

---

## License

MIT
