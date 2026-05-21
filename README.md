# paper-craft-skills

English | [中文](./README.zh.md)

**Turn academic papers into polished method figures and in-depth articles — zero config, one command.**

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="700" alt="Transformer architecture — generated from Attention Is All You Need"/>
</p>

<p align="center">
  From arxiv link to publication-ready visuals and deep-dive articles.<br/>
  Drop a paper, pick a style, get output that looks like a human expert made it.
</p>

---

<table>
<tr>
<td width="50%" align="center" valign="top">

### 🎨 paper-comic
**Paper → Method Figures**

<img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="380"/><br/>
<sub>Transformer architecture — generated from <i>Attention Is All You Need</i></sub>

<br/>

Reads your paper → proposes what to draw → you confirm → generates.

| Style | Vibe |
|-------|------|
| **paper-figure** | Publication-grade diagrams |
| **sketchnote** | Warm hand-drawn research notes |

</td>
<td width="50%" align="center" valign="top">

### 📄 paper-analyzer
**Paper → Deep Articles**

<img src="images/styles_comparison.png" width="380"/><br/>
<sub>Three writing styles: storytelling · academic · concise</sub>

<br/>

Reads the full paper → searches GitHub for code → writes in your chosen style.

| Feature | |
|---------|--|
| 🌐 Output | **HTML** — share anywhere, read on mobile |
| 📐 Formulas | **KaTeX** rendering |
| 📊 Diagrams | **Mermaid** architecture charts |
| ⚡ Setup | **Zero config** — no API keys |

</td>
</tr>
</table>

---

## paper-comic — How it works

```text
/paper-comic https://arxiv.org/abs/1706.03762

Reads the paper, then recommends:

  I suggest 3 figures:
  1. Transformer architecture overview
  2. Self-attention mechanism
  3. Multi-head attention detail

  Language? [Chinese / English]  Style? [sketchnote / paper-figure]  Generate all 3?
```

### Example outputs

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="550"/>
  <br/><b>paper-figure</b> — clean, publication-grade
</p>

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/self-attention-sketchnote.png" width="350"/>
  <br/><b>sketchnote</b> — warm, approachable
</p>

> Full walkthrough: [examples/paper-illustrated/attention-is-all-you-need](./examples/paper-illustrated/attention-is-all-you-need)

---

## paper-analyzer — Three styles, one paper

| Style | Reads like | Use it for |
|-------|-----------|------------|
| **storytelling** | A viral blog post — hooks, analogies, golden takeaway | WeChat, Twitter, blogs |
| **academic** | A peer-reviewed deep dive — KaTeX formulas, comparison tables | Lab meetings, lit reviews |
| **concise** | A cheat sheet — Mermaid diagram + key data table | Quick understanding |

**Also does:** GitHub code search & cross-reference, automatic formula rendering, HTML output ready to share.

---

## Quick Start

```bash
npx skills add zsyggg/paper-craft-skills
```

```bash
/paper-comic https://arxiv.org/abs/1706.03762
/paper-analyzer https://arxiv.org/abs/1706.03762
```

No API keys. No accounts. Image generation uses whatever your environment has — Codex built-in `imagegen`, or any installed backend.

**Works with:** Codex · Claude Code · Cursor · Windsurf

---

MIT
