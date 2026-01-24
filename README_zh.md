# Academic Paper Analyzer - 学术论文深度解析

<div align="center">

[English](README.md) | [中文](README_zh.md)

</div>

> 将学术论文转化为深度技术文章，支持多种写作风格
> 基于 MinerU Cloud API 高精度解析
> **多种写作风格、可选公式讲解、GitHub 代码分析**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)

## 核心特点

### 多种写作风格
| 风格 | 特点 | 适用场景 |
|------|------|----------|
| **storytelling** | 从直觉出发，用比喻讲故事 | 公众号、技术博客、科普 |
| **academic** | 专业术语，严谨表述 | 学术报告、研究组分享 |
| **concise** | 直击核心，信息密度高 | 快速了解、论文速览 |

### 可选增强功能
- **公式讲解**：插入公式图片，详细解读符号含义
- **GitHub 代码分析**：克隆仓库，贴关键源码，代码与论文对照

### 高精度解析
- **MinerU Cloud API**：业界领先的 PDF 解析精度
- **完整提取**：图片、表格、LaTeX 公式
- **智能识别**：自动提取论文结构和元数据

## 快速开始

### 前置准备

1. 安装依赖：
```bash
pip install requests markdown
```

2. 获取 MinerU API Token：
   - 访问 [MinerU 官网](https://mineru.net) 注册
   - 设置环境变量：
     ```bash
     export MINERU_TOKEN="your_token_here"
     ```

### 使用方式

告诉 Claude：

```
请帮我分析这篇论文：/path/to/paper.pdf
```

Claude 会询问你选择：
1. **写作风格**：storytelling / academic / concise
2. **公式讲解**：是 / 否
3. **GitHub 代码分析**：是 / 否（如有开源仓库）

然后自动生成文章。

## 示例

查看 [examples/Engram_Analysis](examples/Engram_Analysis) 了解完整效果。

## 写作风格

### storytelling（默认）

从直觉出发，用比喻讲故事：

```markdown
你有没有想过，大模型其实是没有记忆功能的？每次回答"中国的首都在哪里"，
它都要重新"计算"一遍答案。这不是大炮打蚊子吗？
```

### academic

专业术语，严谨表述：

```markdown
本文提出 Engram，一种基于可扩展查表的条件记忆架构。
该方法将 N-gram 检索与 Transformer 主干结合，
实现 O(1) 复杂度的静态知识获取。
```

### concise

直击核心，信息密度高：

```markdown
**核心创新**：Transformer 中的 O(1) 查表机制
**关键设计**：N-gram 哈希检索 + 上下文门控
**结果**：等参数下 MMLU +3.4，BBH +5.0
```

## 输出格式

| 格式 | 说明 |
|------|------|
| **Markdown**（默认） | 轻量、易编辑 |
| **HTML**（可选） | 图片嵌入，单文件分享 |

## 架构

```
PDF → MinerU API → Markdown + 图片
                        ↓
                风格选择（用户）
                        ↓
                  文章生成
                        ↓
              Markdown / HTML 输出
```

## 脚本

| 脚本 | 用途 |
|------|------|
| `mineru_api.py` | MinerU Cloud API（推荐） |
| `extract_paper_info.py` | 提取论文元数据 |
| `generate_html.py` | Markdown → HTML |

## API 限制

- 单文件最大 200MB
- 最多 600 页
- 支持 PDF、DOC、PPT、图片

## 链接

- [MinerU](https://mineru.net)
- [示例](examples/Engram_Analysis)

## License

MIT
