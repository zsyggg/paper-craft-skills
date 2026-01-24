# paper-craft-skills

[English](./README.md) | 中文

论文工艺：Claude Code 技能集，用于学术论文的深度解读、漫画生成等。

## 安装

### 快速安装（推荐）

```bash
npx skills add zsyggg/paper-craft-skills
```

### 手动安装

告诉 Claude Code：

> 请从 github.com/zsyggg/paper-craft-skills 安装技能

## 可用技能

| 技能 | 描述 | 状态 |
|------|------|------|
| [paper-analyzer](#paper-analyzer) | 将论文转化为多种风格的深度文章 | ✅ 可用 |
| [paper-comic](#paper-comic) | 从论文生成教育漫画 | 🚧 即将推出 |

---

## paper-analyzer

将学术论文转化为深度技术文章，支持多种写作风格。

![Hero Banner](skills/paper-analyzer/images/hero_banner.png)

### 功能特点

| 功能 | 描述 |
|------|------|
| **3 种写作风格** | storytelling（故事型）/ academic（学术型）/ concise（精炼型） |
| **公式讲解** | 插入公式图片并详解符号含义 |
| **代码分析** | 论文概念与 GitHub 源码对照 |
| **高精度解析** | MinerU Cloud API 解析 PDF/图片/表格/LaTeX |

### 使用方法

```
请帮我分析这篇论文：/path/to/paper.pdf
```

Claude 会询问你选择：
1. **风格**：academic（默认）/ storytelling / concise
2. **公式讲解**：是 / 否
3. **代码分析**：是 / 否（如检测到 GitHub 仓库）

### 风格对比

同一篇论文的三种不同风格：

| 风格 | 描述 | 示例 |
|------|------|------|
| **academic** | 正式严谨（默认） | [查看](skills/paper-analyzer/examples/style_comparison/academic.md) |
| storytelling | 故事叙述，生动比喻 | [查看](skills/paper-analyzer/examples/style_comparison/storytelling.md) |
| concise | 表格列表，信息密集 | [查看](skills/paper-analyzer/examples/style_comparison/concise.md) |

### 可选功能

| 功能 | 示例 |
|------|------|
| 公式讲解 | [academic + 公式](skills/paper-analyzer/examples/with_formulas/academic_formulas.md) |
| 代码分析 | [academic + 代码](skills/paper-analyzer/examples/with_code/academic_code.md) |

### 前置准备

```bash
pip install requests markdown
export MINERU_TOKEN="your_token_here"  # 从 https://mineru.net 获取
```

---

## paper-comic

🚧 **即将推出**

从学术论文生成教育漫画，用视觉叙事解释创新点和背景知识。

计划功能：
- 多种漫画风格（Logicomix、漫画指南等）
- 逐格拆解论文概念
- 角色驱动的讲解

---

## License

MIT
