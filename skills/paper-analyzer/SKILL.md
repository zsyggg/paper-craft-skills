---
name: paper-analyzer
description: |
  将学术论文转化为深度技术文章，支持多种写作风格选择。
  使用 MinerU Cloud API 高精度解析 PDF，自动提取图片、表格、公式。
  可选公式讲解、GitHub 代码分析，生成 Markdown 和 HTML 格式。
---

# Academic Paper Analyzer - 学术论文深度解析

## 核心能力

- **MinerU Cloud API** 高精度 PDF 解析
- 自动提取图片、表格、LaTeX 公式
- **多种写作风格**：故事型 / 学术型 / 精炼型
- **可选公式讲解**：插入公式图片并详细解读
- **可选代码分析**：结合 GitHub 开源代码讲解
- 输出 Markdown + HTML（base64 嵌入图片）

## 前置准备

### MinerU API Token

1. 访问 https://mineru.net 注册账号
2. 获取 API Token
3. 设置环境变量（推荐）：
   ```bash
   export MINERU_TOKEN="your_token_here"
   ```

### 依赖安装

```bash
pip install requests markdown
```

## 操作步骤

### 第一步：PDF 解析（使用 MinerU API）

```bash
python scripts/mineru_api.py <pdf_path> <output_dir>
```

或者直接传入 token：
```bash
python scripts/mineru_api.py paper.pdf ./output YOUR_TOKEN
```

**输出结果：**
- `output_dir/*.md` - Markdown 文件（含公式、表格）
- `output_dir/images/` - 高质量提取的图片

### 第二步：提取论文信息

```bash
python scripts/extract_paper_info.py <output_dir>/*.md paper_info.json
```

### 第三步：风格选择（询问用户）

在生成文章前，**必须询问用户**以下选项：

#### 1. 写作风格（必选）

| 风格 | 特点 | 适用场景 |
|------|------|----------|
| **storytelling**（故事型） | 从直觉出发，用比喻和例子，像讲故事 | 公众号、技术博客、科普 |
| **academic**（学术型） | 专业术语，严谨表述，保留原文概念 | 学术报告、论文综述、研究组分享 |
| **concise**（精炼型） | 直击核心，表格列表，信息密度高 | 快速了解、论文速览、技术调研 |

#### 2. 公式选项（可选）

| 选项 | 说明 |
|------|------|
| **with-formulas** | 插入公式图片并详细讲解符号含义 |
| **no-formulas**（默认） | 纯文字描述，不包含公式图片 |

#### 3. 代码选项（可选，仅当论文有 GitHub 时）

| 选项 | 说明 |
|------|------|
| **with-code** | 克隆仓库，贴关键源码，代码与论文对照讲解 |
| **no-code**（默认） | 不包含代码分析 |

**询问示例：**

> 请选择文章风格：
> 1. **academic** - 学术型，专业严谨（默认推荐）
> 2. **storytelling** - 故事型，朴素接地气
> 3. **concise** - 精炼型，快速阅读
>
> 是否需要公式讲解？（论文包含数学公式时推荐）
> 是否需要结合 GitHub 代码分析？（检测到开源仓库：xxx）

**如果用户不确定选哪个，默认使用 academic（学术型）风格。**

### 第四步：智能生成文章

根据用户选择的风格，阅读对应的风格定义文件：
- `styles/storytelling.md` - 故事型风格指南
- `styles/academic.md` - 学术型风格指南
- `styles/concise.md` - 精炼型风格指南
- `styles/with-formulas.md` - 公式讲解指南
- `styles/with-code.md` - 代码分析指南

#### 轻量模式（节省上下文）

**重要：为避免上下文膨胀，请遵循以下原则：**

1. **不要反复读取图片文件** - MinerU 已提取高质量图片，直接引用路径即可
2. **信任 paper_info.json** - 包含图片列表和元数据，无需视觉确认
3. **只看关键图** - 最多读取 1-2 张核心架构图，其余直接引用
4. **让用户验证** - 生成 HTML 后让用户自己检查图片是否正确

#### 通用写作原则

**避免：**
- AI 常用词（"深入探讨"、"至关重要"、"在...领域"）
- 机械化章节标题
- LaTeX 公式语法（如 `$\mathcal{O}(1)$`）- 使用提取的公式图片
- 平铺直叙的技术描述

**采用：**
- 自然段落叙述
- 充分利用 MinerU 提取的图片
- 论文中的每张关键图都应该被讲解到
- 公式截图比 LaTeX 语法更易读

#### storytelling 风格方法论（故事型专用）

以下方法论仅在用户选择 **storytelling** 风格时应用：

**1. 从直觉切入，不要直接讲技术**
- 错误："本文提出了一种基于哈希表的条件记忆模块"
- 正确："你有没有想过，大模型其实是没有记忆功能的？"

**2. 先讲历史背景，再讲创新**
- 介绍新技术前，先解释相关的旧技术
- 让读者理解"为什么需要这个创新"

**3. 用简单例子贯穿全文**
- 选一个简单的例子反复使用
- 例如："中国的首都在北京"

**4. 使用生动的比喻**
- "大炮打蚊子"、"查字典 vs 背字典"
- 让抽象概念具象化

**5. 逻辑递进，层层深入**
- 简单问题 → 复杂问题 → 解决方案

**6. 提炼核心洞见**
- 用一句话总结，如"记忆归记忆，计算归计算"

#### 文章结构

**1. 论文信息**

```markdown
**论文标题**：xxx
**论文链接**：[arXiv](https://arxiv.org/abs/xxxx)
**作者团队**：xxx
```

**2. 直觉引入**（2-3段）
- 从一个问题或场景开始
- 让读者产生好奇心
- 引出"为什么需要这个研究"

**3. 背景知识**（3-4段）
- 解释相关的基础技术或历史方法
- 用简单例子说明
- 让读者理解现有方案的局限

**4. 核心创新**（4-5段）
- 详细讲解论文的创新点
- 每个创新点都要有图片支撑
- 用比喻和例子让抽象概念具象化
- 公式用图片展示，不用 LaTeX 语法

**5. 实验验证**（2-3段）
- 关键的实验结果图表
- 对比分析和数据解读
- 突出最亮眼的结果

**6. 深入分析**（2-3段）
- 机制分析、消融实验等
- 解释"为什么这个方法有效"
- 提供更深层次的理解

**7. 思考与展望**（1-2段）
- 提炼核心洞见
- 预测未来发展方向
- 个人观点和评价

### 第五步：输出格式（询问用户）

**默认输出 Markdown**，文章写完后询问用户是否需要其他格式：

> "文章已生成：`article.md`。需要生成 HTML 版本吗？（HTML 会嵌入图片，方便直接分享）"

**格式对比：**
| 格式 | 优势 | 适用场景 |
|------|------|----------|
| MD（默认） | 轻量、易编辑、公众号可直接导入 | 日常使用 |
| HTML | 图片嵌入、单文件分享 | 预览效果、分享给他人 |

如果用户需要 HTML：
```bash
python scripts/generate_html.py <article.md> <output.html>
```

## 资源索引

**风格定义：**
- `styles/storytelling.md` - 故事型风格
- `styles/academic.md` - 学术型风格
- `styles/concise.md` - 精炼型风格
- `styles/with-formulas.md` - 公式讲解
- `styles/with-code.md` - 代码分析

**脚本：**
- `scripts/mineru_api.py` - MinerU Cloud API 调用（推荐）
- `scripts/convert_pdf.py` - 本地转换（备选，需要 PyMuPDF）
- `scripts/extract_paper_info.py` - 提取论文元数据
- `scripts/generate_html.py` - 生成 HTML（base64 图片）

## 注意事项

- **优先使用 MinerU API**，精度最高，支持公式/表格
- **节省上下文**：不要反复读取图片，信任元数据
- 不输出分析过程，用户只看最终文章
- 避免分点列表，使用自然段落叙述
- 图片选择 3-5 张关键图表

## API 限制

- 单个文件最大 200MB
- 单个文件最多 600 页
- 支持 PDF、DOC、PPT、图片等格式
