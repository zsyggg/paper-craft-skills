# paper-craft-skills

English | [中文](./README.zh.md)

**8 AI-powered skills for academic papers: figures, articles, slides, surveys, posters, summaries, mind maps & reviews — zero config, one command.**

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="700" alt="Transformer architecture"/>
</p>

<p align="center">
  From arxiv link to publication-ready visuals, AIGC slide decks, deep articles, surveys, posters, summaries, mind maps, and peer reviews.<br/>
  Drop a paper, pick a style, get output that looks like a human expert made it.
</p>

[![CI](https://github.com/zsyggg/paper-craft-skills/actions/workflows/quality-check.yml/badge.svg)](https://github.com/zsyggg/paper-craft-skills/actions/workflows/quality-check.yml)

---

## How to install

**Copy this into Codex or Claude Code:**

```
Please install zsyggg/paper-craft-skills for me.
GitHub: https://github.com/zsyggg/paper-craft-skills
```

**No API keys. No accounts.** Or via terminal:

```bash
npx skills add zsyggg/paper-craft-skills
```

**Works with:** Codex · Claude Code · Cursor · Windsurf

---

## 8 Skills at a Glance

| Skill | Action | Output |
|-------|--------|--------|
| 🎨 **paper-comic** | Paper → Method Figures | AIGC diagrams (PNG) |
| 📄 **paper-analyzer** | Paper → Deep Articles | HTML with KaTeX + Mermaid |
| 🖼️ **paper-deck** | Paper → Visual Slide Deck | PPTX + PDF (4 styles) |
| 🔬 **paper-survey** | Multi-Paper → Comparison Survey | Figures/Slides/Article |
| 📋 **paper-poster** | Paper → Conference Poster | 300dpi printable PDF |
| 📊 **paper-summary** | Paper → One-Page Visual Summary | Infographic/Card/Timeline |
| 🧠 **paper-mindmap** | Paper → Mind Map | Mermaid code / AIGC image |
| 📝 **paper-review** | Paper → Peer Review | 3 reviewer reports + decision |

---

## Quick Reference

```bash
# Method Figures
/paper-comic https://arxiv.org/abs/1706.03762 --style sketchnote

# Deep Analysis
/paper-analyzer paper.pdf --style academic

# Slides
/paper-deck paper.pdf --style journal-minimal --slides 12

# Multi-Paper Comparison
/paper-survey p1.pdf p2.pdf --format survey-article

# Conference Poster
/paper-poster paper.pdf --style conference-wide

# Visual Summary
/paper-summary paper.pdf --mode infographic

# Mind Map
/paper-mindmap paper.pdf

# Peer Review
/paper-review paper.pdf
```

---

## Unified CLI

```bash
python3 scripts/papercraft.py list
python3 scripts/papercraft.py comic paper.pdf --style sketchnote
python3 scripts/papercraft.py survey p1.pdf p2.pdf --format survey-article
python3 scripts/papercraft.py poster paper.pdf --style defense-poster
python3 scripts/papercraft.py summary paper.pdf --mode infographic
```

---

## Batch Generation Dashboard

Generate all example prompts at once and open an interactive HTML dashboard:

```bash
python3 scripts/batch_generate.py --html
# Opens generation-dashboard.html — click to copy any prompt
python3 scripts/batch_generate.py --list      # List all available prompts
python3 scripts/batch_generate.py poster       # Just poster prompts
```

---

## Examples

- [Transformer Illustrated — paper-comic](./examples/paper-illustrated/attention-is-all-you-need)
- [Transformer Infographic — paper-summary](./examples/paper-summary/attention-is-all-you-need)
- [Conference Poster — paper-poster](./examples/paper-poster/attention-is-all-you-need)
- [Position Encoding Comparison — paper-survey](./examples/paper-survey/transformer-position-encoding)
- [Deck Quality Check — paper-deck](./examples/paper-deck-quality-check)

---

## tech stack

```bash
pip install Pillow python-pptx
# Optional: markdown pypdf2 for paper-analyzer
```

---

## License

MIT

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=zsyggg/paper-craft-skills&type=Date)](https://star-history.com/#zsyggg/paper-craft-skills&Date)
