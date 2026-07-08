---
name: paper-slides-html
description: |
  将论文生成为 HTML/CSS 幻灯片（reveal.js 风格），不需要图片生成后端。
  纯前端、零依赖、可编辑、可直接分享。适合没有生图环境时的替代方案。
  支持学术风、极简风、深色模式三种主题。
  当用户提到"HTML 幻灯片""网页 PPT""浏览器演示""不用生图的 PPT""可编辑幻灯片"时使用。
---

# Paper Slides HTML — Browser-Based Presentations

把论文内容组织成纯 HTML/CSS 幻灯片，零依赖、可直接分享。

## 核心哲学

### HTML 幻灯片 vs AIGC 幻灯片

| | paper-slides-html | paper-deck (AIGC) |
|---|---|---|
| 依赖 | 无（纯 HTML/CSS） | 需要生图后端 |
| 可编辑性 | 完全可编辑（改 HTML） | 改 prompt 重生成 |
| 文件大小 | ~50KB | ~50MB (图片) |
| 视觉效果 | 干净、现代 CSS 设计 | 无限风格可能 |
| 适合场景 | 快速出稿、协作编辑 | 高质感汇报、传播 |

### 使用场景

- 没有生图后端可用时的降级方案
- 需要多人协作编辑幻灯片内容
- 快速出稿给导师/同学看内容再美化
- 嵌入网页、作为在线演示

---

## 三种主题

| 主题 | 风格 | 适合 |
|------|------|------|
| **academic** | 学术蓝白风，Serif 字体 | 论文组会、答辩 |
| **minimal** | 极简黑白风，Sans-serif | 技术分享、内部讨论 |
| **dark** | 深色模式，科技感 | 演示、公开分享 |

---

## 幻灯片结构模板

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>论文标题 — 幻灯片</title>
<style>
  /* 主题样式自动生成 */
  /* 支持 academic / minimal / dark 三种主题 */
</style>
</head>
<body>
<div class="slides">

  <section class="slide cover">
    <h1>论文标题</h1>
    <p class="subtitle">作者 · 会议/年份</p>
    <p class="affiliation">单位</p>
  </section>

  <section class="slide">
    <h2>目录</h2>
    <ol>
      <li>研究背景与问题</li>
      <li>方法概述</li>
      <li>核心机制</li>
      <li>实验结果</li>
      <li>结论与贡献</li>
    </ol>
  </section>

  <section class="slide">
    <h2>研究问题</h2>
    <div class="content">
      <div class="problem-box">
        <strong>现有局限：</strong>
        <p>...</p>
      </div>
      <div class="goal-box">
        <strong>本文目标：</strong>
        <p>...</p>
      </div>
    </div>
  </section>

  <section class="slide">
    <h2>方法总览</h2>
    <div class="method-flow">
      <!-- 方法流程图用纯 CSS 绘制 -->
      <div class="flow-step">输入</div>
      <div class="flow-arrow">→</div>
      <div class="flow-step highlight">核心模块</div>
      <div class="flow-arrow">→</div>
      <div class="flow-step">输出</div>
    </div>
    <p class="caption">图：方法总览流程</p>
  </section>

  <!-- 更多页面... -->

  <section class="slide">
    <h2>结论</h2>
    <ul class="takeaway">
      <li>贡献 1</li>
      <li>贡献 2</li>
      <li>贡献 3</li>
    </ul>
  </section>

</div>
<script>
  // 键盘导航：左右键翻页
  // 按 F 全屏
</script>
</body>
</html>
```

## 工作流

### Step 1: 分析论文

同 paper-analyzer/paper-deck 的分析方式：读论文、提取核心内容。

### Step 2: 确认

| 确认项 | 选项 |
|--------|------|
| 主题 | academic / minimal / dark |
| 页数 | 8-15 页（默认 10 页） |
| 语言 | 中文 / English |
| 是否含公式 | 是/否 |
| 是否含代码段 | 是/否 |

### Step 3: 生成 HTML

按论文内容和确认结果，生成完整自包含的 HTML 幻灯片文件。

### Step 4: 输出

- `slides.html` — 完整幻灯片（可直接浏览器打开）
- 或 `slides.pdf` — 通过浏览器打印为 PDF

---

## 页面类型

| 类型 | 内容 | 页数建议 |
|------|------|---------|
| cover | 标题、作者、单位 | 1 |
| toc | 目录 | 1 |
| problem | 研究问题与动机 | 1-2 |
| method | 方法流程与核心机制 | 2-4 |
| experiments | 实验设置与结果 | 1-3 |
| contribution | 贡献总结 | 1 |
| thanks | 致谢/参考 | 1 |

## 每页设计原则

- 每页只讲一个观点
- 标题 28-36px，正文 18-22px
- 代码段用等宽字体 + 代码高亮
- 表格干净简约，无多余边框
- 使用 admonition 块（提示/注意/要点）
- 方法流程使用纯 CSS flexbox 布局

## 参考文件

- `themes/academic.css` — 学术主题样式
- `themes/minimal.css` — 极简主题样式
- `themes/dark.css` — 深色主题样式
- `scripts/extract_to_slides.py` — 论文→幻灯片 JSON 提取脚本
