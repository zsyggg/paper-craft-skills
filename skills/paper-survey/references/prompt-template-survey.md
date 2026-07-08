# Prompt Template — 对比图生成

对比图和单论文图的核心区别：**同一页内要呈现多篇论文的并列信息**。

## 模板

```markdown
---
slide: NN
title: "对比维度标题"
role: "comparison-overview | dimension-comparison | timeline | tradeoff"
style_preset: "journal-minimal"
language: "zh"
aspect_ratio: "16:9"
output: "images/NN-comparison-slug.png"
---

Create one complete 16:9 presentation slide image for a multi-paper comparison.

Purpose:
[这页要回答的对比问题：哪个维度、几篇论文、什么差异]

Papers:
- Paper A (blue): [one-line method label]
- Paper B (orange): [one-line method label]
- Paper C (green): [one-line method label]

Style:
[Use the selected preset phrase. Add "comparison layout" modifier.]

Composition:
[Describe the spatial layout:]
- [Number of columns/rows, how papers are positioned]
- [Where the shared structure goes, where differences go]
- [Highlight regions for key differences]

Content:
- Shared structure: [elements all papers have in common]
- Difference 1: [what Paper A does vs Paper B vs Paper C]
- Difference 2: [another dimension]
- Key annotation: [the single most important takeaway from this comparison]

Color Rules:
- Paper A: blue (#2563eb)
- Paper B: orange (#ea580c)
- Paper C: green (#16a34a)
- Shared/common elements: neutral gray (#6b7280)
- Difference highlights: use dashed borders, arrows, or highlight boxes

Visual Details:
- [Line style, card style, arrow style]
- [How to indicate same/different/evolved]
- [Data visualization style if showing metrics]

Constraints:
- Keep all text short and readable (labels ≤ 10 characters).
- Each paper's section must be visually distinct but comparable.
- Use consistent visual language across all papers (same shape language, same label style).
- Do not generate page numbers, logos, watermarks.
- The comparison must be immediately obvious at a glance.
```

## 常用构图模式

### 左右对比（2 篇论文）

```
┌──────────────────────────────────────┐
│            对比标题                    │
├──────────────────┬───────────────────┤
│    Paper A       │    Paper B        │
│  ┌────────────┐  │  ┌─────────────┐  │
│  │ 方法流程   │  │  │ 方法流程    │  │
│  │ (蓝色)     │  │  │ (橙色)      │  │
│  └────────────┘  │  └─────────────┘  │
│  关键差异标注 ←──┼──→ 关键差异标注   │
├──────────────────┴───────────────────┤
│              Bottom annotation       │
└──────────────────────────────────────┘
```

### 三分对比（3 篇论文）

```
┌──────────────────────────────────────────┐
│              对比标题                      │
├──────────┬───────────┬───────────────────┤
│ Paper A  │ Paper B   │ Paper C           │
│ (蓝色)   │ (橙色)    │ (绿色)            │
│ 流程     │ 流程      │ 流程              │
│ 指标: X  │ 指标: Y   │ 指标: Z           │
├──────────┴───────────┴───────────────────┤
│             差异点总结                     │
└──────────────────────────────────────────┘
```

### 演进时间线

```
┌──────────────────────────────────────────┐
│           研究演进时间线                    │
│                                           │
│  2017         2019         2021    2023   │
│   │            │            │       │     │
│ Paper A ──→ Paper B ──→ Paper C ──→ ...  │
│  (idea)    (改进X)    (突破Y)   (应用)    │
│   │            │            │       │     │
│   └───── 关键标注 ──────────┘            │
└──────────────────────────────────────────┘
```

### 实验对比

```
┌──────────────────────────────────────────┐
│            Benchmark 对比                  │
├──────────┬────────┬────────┬─────────────┤
│ Dataset  │ Paper A│ Paper B│ Paper C     │
├──────────┼────────┼────────┼─────────────┤
│ Bench-1  │ 85.2   │ 87.1   │ 89.3        │
│ Bench-2  │ 62.4   │ 65.8   │ 71.2        │
│ Speed    │ 1.0x   │ 1.2x   │ 0.8x        │
├──────────┴────────┴────────┴─────────────┤
│ Best in each: Paper A (speed), Paper B    │
│ (balance), Paper C (accuracy)             │
└──────────────────────────────────────────┘
```

## 对比视觉设计手册

### 颜色分配（3 篇以内）
- Paper A: 蓝色 #2563eb
- Paper B: 橙色 #ea580c
- Paper C: 绿色 #16a34a
- 公共/相同: 中性灰 #6b7280

### 颜色分配（4-6 篇）
- 使用色轮均匀分布：蓝、橙、绿、紫、红、青
- 每篇论文一个专属色，所有图中保持不变

### 相同 vs 差异的视觉表达

| 关系 | 视觉表达 |
|------|---------|
| 相同/共享 | 灰色 + 实线框 |
| 差异 | 各论文颜色 + 虚线框 + 差异箭头标注 |
| 演进 | 箭头 + 时间线 + "v1 → v2" 标注 |

### 每页信息密度

| 论文数 | 推荐每页对比维度数 | 说明 |
|--------|------------------|------|
| 2 篇 | 1-2 个维度 | 每页聚焦，充分展示差异 |
| 3 篇 | 1 个维度 | 三分布局每页只比一个维度 |
| 4+ 篇 | 1 个维度 | 网格布局，用颜色区分论文 |
