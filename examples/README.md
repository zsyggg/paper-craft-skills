# Examples - 示例说明

本目录展示了 Academic Paper Analyzer 的各种输出示例，均基于同一篇论文：

**示例论文**：[Engram: Conditional Memory via Scalable Lookup](https://github.com/deepseek-ai/Engram)
- **作者**：北京大学 & DeepSeek-AI
- **主题**：为 Transformer 引入 O(1) 查表机制，分离静态知识与动态推理

---

## 目录结构

```
examples/
├── README.md                 # 本文件
├── style_comparison/         # 三种写作风格对比
│   ├── storytelling.md       # 故事型：比喻、直觉、像聊天
│   ├── academic.md           # 学术型：专业、严谨、结构化（默认推荐）
│   ├── concise.md            # 精炼型：表格、列表、信息密集
│   └── images/               # 论文图片
├── with_formulas/            # 公式讲解功能示例
│   └── academic_formulas.md  # 学术风格 + 公式详解
└── with_code/                # GitHub 代码分析功能示例
    └── academic_code.md      # 学术风格 + 代码对照
```

---

## 示例预览

### 1. 三种写作风格对比

同一篇论文，三种不同表达：

| 风格 | 特点 | 适合读者 | 字数 |
|------|------|----------|------|
| [storytelling](style_comparison/storytelling.md) | 从直觉出发，用比喻讲故事 | 技术博客、公众号、科普 | ~4000 |
| [**academic**](style_comparison/academic.md) | 专业术语，严谨结构 | 研究报告、论文综述 | ~3500 |
| [concise](style_comparison/concise.md) | 表格列表，信息密集 | 快速了解、技术调研 | ~2000 |

**推荐**：如果不确定选哪个，选 **academic**（学术型）最稳妥。

### 2. 公式讲解功能

[with_formulas/academic_formulas.md](with_formulas/academic_formulas.md)

展示如何在文章中插入公式图片并详细解读：
- 插入论文中的公式截图
- 逐个解释符号含义
- 连接公式与直观理解

### 3. GitHub 代码分析功能

[with_code/academic_code.md](with_code/academic_code.md)

展示如何结合开源代码讲解论文：
- 自动克隆 GitHub 仓库
- 提取关键实现代码
- 代码与论文概念对照

---

## 如何选择？

```
你需要什么？
    │
    ├─ 给普通读者看 ──────────▶ storytelling（故事型）
    │
    ├─ 给专业人士看 ──────────▶ academic（学术型）⭐ 默认
    │
    ├─ 快速了解论文 ──────────▶ concise（精炼型）
    │
    ├─ 需要讲解数学公式 ──────▶ + with_formulas
    │
    └─ 论文有开源代码 ────────▶ + with_code
```

风格和增强功能可以组合，例如：
- `storytelling + with_formulas`：故事型 + 公式讲解
- `academic + with_code`：学术型 + 代码分析
- `concise + with_formulas + with_code`：精炼型 + 公式 + 代码

---

## 生成自己的文章

```
请分析这篇论文：/path/to/paper.pdf
```

Claude 会询问你选择风格和增强功能，然后自动生成文章。
