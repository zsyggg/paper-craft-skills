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
| **sketchnote** | 温暖手绘研究笔记 |

</td>
<td width="50%" align="center" valign="top">

### 📄 paper-analyzer
**论文 → 深度长文**

<img src="images/styles_comparison.png" width="380"/><br/>
<sub>三种写作风格：故事型 · 学术型 · 精炼型</sub>

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

  建议生成 3 张图：
  1. Transformer 架构总览
  2. Self-attention 机制（核心创新）
  3. Multi-head attention 细节

  语言？[中文 / English]  风格？[sketchnote / paper-figure]  生成全部 3 张？
```

### 实际效果

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/transformer-overview-paper-figure.png" width="550"/>
  <br/><b>paper-figure</b> — 论文级别专业图表
</p>

<p align="center">
  <img src="examples/paper-illustrated/attention-is-all-you-need/self-attention-sketchnote.png" width="350"/>
  <br/><b>sketchnote</b> — 温暖手绘风格
</p>

> 完整示例：[examples/paper-illustrated/attention-is-all-you-need](./examples/paper-illustrated/attention-is-all-you-need)

---

## paper-analyzer — 一篇论文，三种写法

| 风格 | 读起来像 | 适合 |
|------|---------|------|
| **storytelling** | 公众号爆文 — 钩子开头、类比贯穿、金句收尾 | 公众号、推特、技术博客 |
| **academic** | 学术综述 — KaTeX 公式、对比表格、深度分析 | 组会分享、文献综述 |
| **concise** | 速查表 — Mermaid 流程图 + 关键数据表 | 快速了解、预读梳理 |

**还能做的事：** 自动搜索 GitHub 开源代码、对照论文讲解、HTML 输出可直接分享。

---

## 快速开始

```bash
npx skills add zsyggg/paper-craft-skills
```

```bash
/paper-comic https://arxiv.org/abs/1706.03762
/paper-analyzer https://arxiv.org/abs/1706.03762
```

不需要任何 API key。不需要注册。图片生成自动使用你环境里已有的能力——Codex 内置 imagegen 或其他已安装的后端。

**支持：** Codex · Claude Code · Cursor · Windsurf

---

MIT
