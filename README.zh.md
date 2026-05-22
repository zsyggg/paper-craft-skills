# paper-craft-skills

[English](./README.md) | 中文

**把学术论文变成精美的方法图解和深度长文。零配置，一行命令。**

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="700" alt="基于 Attention Is All You Need 生成的 Transformer 架构图"/>
</p>

<p align="center">
  输入 arxiv 链接，选择风格，输出像人类专家手笔的图解和文章。
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

## 两个技能

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
</table>

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

MIT
