# paper-craft-skills

[English](./README.md) | 中文

**把学术论文变成精美的方法图解、高质感 AIGC 幻灯片、深度长文和多论文对比综述。零配置，一行命令。**

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="700" alt="基于 Attention Is All You Need 生成的 Transformer 架构图"/>
</p>

<p align="center">
  输入 arxiv 链接，选择风格，输出像人类专家手笔的图解、PPT、文章和综述。
</p>

---

## 怎么安装

**把这段话复制到 Codex 或 Claude Code 里：**

```
请帮我安装 zsyggg/paper-craft-skills
GitHub：https://github.com/zsyggg/paper-craft-skills
```

AI agent 会自动克隆、配置、注册 skill。**不需要 API key，不需要注册账号。** 如果想用终端：

```bash
npx skills add zsyggg/paper-craft-skills
```

**支持：** Codex · Claude Code · Cursor · Windsurf

---

## 四个技能

<table>
<tr>
<td width="50%" align="center" valign="top">

### 🎨 paper-comic
**论文 → 方法图解**

<img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="380"/><br/>
<sub>Transformer 架构图 — 基于 <i>Attention Is All You Need</i> 生成</sub>

<br/>

读完论文 → 推荐画什么 → 你确认 → 生成。

| 风格 | 效果 |
|------|------|
| **paper-figure** | 论文级别专业图表 |
| **sketchnote** | 明亮温暖的手抄报式研究笔记 |

</td>
<td width="50%" align="center" valign="top">

### 📄 paper-analyzer
**论文 → 深度长文**

<img src="images/hero_banner.png" width="380"/><br/>
<sub>论文 → 精美排版 HTML 文章，含公式、代码对照、多种风格</sub>

<br/>

读完论文全文 → 搜索 GitHub 源码 → 按你选的风格写作。

| 特性 | |
|------|--|
| 🌐 输出 | **HTML** — 手机/桌面都能看 |
| 📐 公式 | **KaTeX** 渲染 |
| 📊 图表 | **Mermaid** 架构图 |
| ⚡ 配置 | **零配置** — 不需要任何 API key |

</td>
</tr>
<tr>
<td width="50%" align="center" valign="top">

### 🖼️ paper-deck
**论文 → 视觉幻灯片**

<img src="images/paper_deck_intro.png" width="380"/><br/>
<sub>论文/文章 → 分析 → 大纲 → 视觉提示词 → PPTX/PDF</sub>

<br/>

读完论文 → 规划 deck → 生成 slide image → 导出 PPTX/PDF。

| 输出 | |
|------|--|
| 🎞️ 幻灯片 | 16:9 视觉页面 |
| 📦 导出 | `.pptx` + `.pdf` |
| 🛠️ 返修 | 任意页可重生成 |

</td>
<td width="50%" align="center" valign="top">

### 🔬 paper-survey
**多论文 → 对比综述** ✨ 全新

<img src="images/hero_banner.png" width="380"/><br/>
<sub>多篇论文 → 对比图解、综述幻灯片或深度对比长文</sub>

<br/>

逐篇分析 → 构建对比矩阵 → 三模式输出。

| 模式 | 输出 |
|------|------|
| 🎨 comparison-figures | AIGC 对比图解 (PNG) |
| 🎞️ survey-deck | 对比幻灯片 (PPTX/PDF) |
| 📄 survey-article | 深度对比长文 (HTML) |

</td>
</tr>
</table>

---

## paper-deck — 不像模板的视觉幻灯片

`paper-deck` 可以把论文、文章或技术笔记变成一套有设计感的幻灯片。它会先生成 deck brief 和逐页大纲，再写可复现的视觉提示词，逐页生成 16:9 slide image，最后合成为 `.pptx` 和 `.pdf`。

它很适合细节返修：每一页都有独立 prompt，所以你可以继续要求"第 5 页更像论文图""第 8 页换成真实 benchmark 图""保留布局但把封面换成玻璃质感"。

<p align="center">
  <img src="images/paper_deck_styles.png" width="640"/>
  <br/><sub>同一个论文主题的四种紧凑风格预览</sub>
</p>

| 风格 | 适合 |
|------|------|
| **journal-minimal** | Nature/IEEE 风学术汇报、答辩、组会 |
| **business-research** | 商业研究、行业分析、投资人/客户汇报 |
| **warm-notes** | 温暖手记风讲解、课程、论文学习笔记 |
| **liquid-glass** | Apple 式玻璃质感封面、章节页和高冲击视觉页 |

它也支持真实素材。如果 PDF 里有好的论文图、表格、实验曲线或截图，skill 会先规划哪些页面应该使用真实素材、怎么裁切、放在第几页、用什么框架承载。

```bash
/paper-deck https://arxiv.org/abs/1706.03762
/paper-deck /path/to/paper.pdf --style journal-minimal --slides 12
/paper-deck notes.md --style liquid-glass
```

---

## paper-comic — 怎么用的

```text
/paper-comic https://arxiv.org/abs/1706.03762

读完论文，会推荐：

  建议生成 6 张图：
  1. 封面图：一句话贡献 + 视觉锚点
  2. Transformer 架构总览
  3. Self-attention 机制
  4. Multi-head attention 细节
  5. Encoder / Decoder Block
  6. 关键结果

  也可以只生成 1 张总览图，或扩展到 8 张把机制讲更细。
  语言？[中文 / English]  风格？[sketchnote / paper-figure]  范围和张数？
```

### 实际效果

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="550"/>
  <br/><b>paper-figure</b> — 论文级别专业图表
</p>

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/self-attention-sketchnote.png" width="350"/>
  <br/><b>sketchnote</b> — 明亮温暖的手抄报风格
</p>

> 完整示例：[examples/paper-illustrated/attention-is-all-you-need](./examples/paper-illustrated/attention-is-all-you-need)

---

## paper-analyzer — 一篇论文，三种写法

**不是翻译论文，是重新讲给你听。** 读完论文全文、搜索 GitHub 开源代码、对照论文讲解，按你选的风格写成高质量文章。

### 三种写作风格

<p align="center">
  <img src="images/styles_comparison.png" width="700"/>
</p>

| 风格 | 读起来像 | 适合 |
|------|---------|------|
| **storytelling** | 公众号爆文 — 钩子开头、类比贯穿、金句收尾 | 公众号、推特、技术博客 |
| **academic** | 学术综述 — KaTeX 公式、对比表格、深度分析 | 组会分享、文献综述 |
| **concise** | 速查表 — Mermaid 流程图 + 关键数据表 | 快速了解、预读梳理 |

### 功能展示

<table>
<tr>
<td width="50%" align="center">
<img src="images/formula_feature.png" width="360"/><br/>
<b>公式讲解</b><br/>
提取论文公式，逐个符号拆解含义
</td>
<td width="50%" align="center">
<img src="images/code_feature.png" width="360"/><br/>
<b>代码对照</b><br/>
论文方法 + GitHub 源码，一一对应讲解
</td>
</tr>
</table>

```bash
/paper-analyzer https://arxiv.org/abs/1706.03762     # arxiv 链接
/paper-analyzer /path/to/paper.pdf                     # 本地 PDF
/paper-analyzer                                         # 然后粘贴文本
```

---

## paper-survey — 多论文对比综述

`paper-survey` 把 paper-craft-skills 的能力延伸到**多篇论文**。它会逐篇独立分析、构建结构化对比矩阵（5 维度：问题定义、方法路径、关键设计、实验表现、适用边界），然后三模式输出：

| 模式 | 输出 | 适合 |
|------|------|------|
| **comparison-figures** | AIGC 对比图解 (PNG) | 快速看清差异、技术选型 |
| **survey-deck** | 16:9 对比幻灯片 (PPTX/PDF) | 组会综述报告 |
| **survey-article** | 深度对比长文 (HTML) | 文献综述写作 |

每篇论文在所有输出中使用固定颜色，差异点视觉高亮，对比矩阵确保评估维度一致。

```bash
/paper-survey https://arxiv.org/abs/1706.03762 https://arxiv.org/abs/1907.04340
/paper-survey --topic "Transformer 位置编码对比" --papers p1.pdf p2.pdf p3.pdf
/paper-survey notes.md --format article
```

---

MIT
