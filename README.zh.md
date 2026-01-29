# paper-craft-skills

[English](./README.md) | 中文

论文工艺：Claude Code 技能集，用于学术论文的深度解读、漫画生成等。

**兼容**：Cursor、Codex、Windsurf 等支持 Claude Code 技能的 AI 编程助手。

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
| [paper-comic](#paper-comic) | 从论文生成教育漫画 | ✅ 可用 |
| [baoyu-gemini-web](#baoyu-gemini-web) | 使用 Gemini Web 的图片生成后端 | ✅ 可用 |

---

## paper-analyzer

将学术论文转化为深度技术文章，支持多种写作风格。

![Hero Banner](images/hero_banner.png)

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

![风格对比](images/styles_comparison.png)

| 风格 | 描述 | 示例 |
|------|------|------|
| **academic** | 正式严谨（默认） | [查看](examples/style_comparison/academic.md) |
| storytelling | 故事叙述，生动比喻 | [查看](examples/style_comparison/storytelling.md) |
| concise | 表格列表，信息密集 | [查看](examples/style_comparison/concise.md) |

### 可选功能

#### 公式讲解

插入公式图片，详解每个符号含义。

![公式功能](images/formula_feature.png)

→ [查看示例：academic + 公式](examples/with_formulas/academic_formulas.md)

#### 代码分析

将论文概念与 GitHub 源码对照。

![代码功能](images/code_feature.png)

→ [查看示例：academic + 代码](examples/with_code/academic_code.md)

### 前置准备

```bash
pip install requests markdown
export MINERU_TOKEN="your_token_here"  # 从 https://mineru.net 获取
```

---

## paper-comic

从学术论文生成教育漫画，用视觉叙事解释核心概念。

### 前置准备

- **Google 账号**：用于 Gemini Web 认证
- **baoyu-gemini-web**：图片生成后端（已包含在本包中）
- **Bun 运行时**：`npm install -g bun` 或使用 `npx -y bun`

首次运行会打开 Chrome 浏览器进行 Google 账号认证，之后 Cookie 会被缓存。

### 功能特点

| 功能 | 描述 |
|------|------|
| **4 种画风** | classic / tech / warm / chalk |
| **10 页漫画** | 完整叙事，角色驱动 |
| **视觉比喻** | 抽象概念具象化 |
| **中文对话** | 所有文字使用中文 |

### 画风选择

<table>
<tr>
<td align="center" width="25%"><img src="examples/paper-comic/cover-classic.png" width="180"/><br/><b>classic</b><br/>通用论文</td>
<td align="center" width="25%"><img src="examples/paper-comic/cover-tech.png" width="180"/><br/><b>tech</b><br/>AI/计算机</td>
<td align="center" width="25%"><img src="examples/paper-comic/cover-warm.png" width="180"/><br/><b>warm</b><br/>心理学/教育</td>
<td align="center" width="25%"><img src="examples/paper-comic/cover-chalk.png" width="180"/><br/><b>chalk</b><br/>数学/物理</td>
</tr>
</table>

### 示例：Engram 论文（10 页）

<table>
<tr>
<td align="center"><img src="examples/paper-comic/00-cover.png" width="120"/><br/>封面</td>
<td align="center"><img src="examples/paper-comic/01-page.png" width="120"/><br/>第1页</td>
<td align="center"><img src="examples/paper-comic/02-page.png" width="120"/><br/>第2页</td>
<td align="center"><img src="examples/paper-comic/03-page.png" width="120"/><br/>第3页</td>
<td align="center"><img src="examples/paper-comic/04-page.png" width="120"/><br/>第4页</td>
</tr>
<tr>
<td align="center"><img src="examples/paper-comic/05-page.png" width="120"/><br/>第5页</td>
<td align="center"><img src="examples/paper-comic/06-page.png" width="120"/><br/>第6页</td>
<td align="center"><img src="examples/paper-comic/07-page.png" width="120"/><br/>第7页</td>
<td align="center"><img src="examples/paper-comic/08-page.png" width="120"/><br/>第8页</td>
<td align="center"><img src="examples/paper-comic/09-page.png" width="120"/><br/>第9页</td>
</tr>
<tr>
<td align="center"><img src="examples/paper-comic/10-page.png" width="120"/><br/>第10页</td>
<td></td>
<td></td>
<td></td>
<td></td>
</tr>
</table>

---

## baoyu-gemini-web

使用 Google Gemini Web 的图片生成后端。paper-comic 使用此技能生成漫画页面。

### 功能特点

| 功能 | 描述 |
|------|------|
| **图片生成** | 从文本提示生成图片 |
| **多轮对话** | 使用 sessionId 保持上下文 |
| **文本生成** | 也支持文本响应 |

### 前置准备

- **Google 账号**：用于认证
- **Chrome 浏览器**：用于首次登录
- **Bun 运行时**：`npm install -g bun`

### 使用方法

```bash
# 文本生成
npx -y bun scripts/main.ts "你的提示词"

# 图片生成
npx -y bun scripts/main.ts "一个可爱的机器人" --image robot.png

# 多轮对话（用于保持角色一致性）
npx -y bun scripts/main.ts "设计一个角色" --sessionId my-session
npx -y bun scripts/main.ts "画同一个角色在挥手" --sessionId my-session
```

### 认证

首次运行会打开 Chrome 进行 Google 账号认证，之后 Cookie 会被缓存。

```bash
# 强制刷新 Cookie
npx -y bun scripts/main.ts --login
```

---

## License

MIT
