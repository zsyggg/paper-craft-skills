# paper-craft-skills

[English](./README.md) | 中文

**一键把学术论文变成深度解读文章或教育漫画**

<table>
<tr>
<td width="50%" align="center">
<b>paper-analyzer</b><br/>
论文 → 深度文章<br/><br/>
<img src="images/hero_banner.png" width="400"/>
</td>
<td width="50%" align="center">
<b>paper-comic</b><br/>
论文 → 10页漫画<br/><br/>
<img src="examples/paper-comic/00-cover.png" width="200"/>
</td>
</tr>
</table>

## 快速开始

### 1. 安装

```bash
npx skills add zsyggg/paper-craft-skills
```

### 2. 配置 API

| 技能 | 需要配置 |
|------|----------|
| paper-analyzer | `MINERU_TOKEN` - 从 [mineru.net](https://mineru.net) 获取 |
| paper-comic | Google 账号（首次运行自动弹出登录） |

```bash
# paper-analyzer 需要
pip install requests markdown
export MINERU_TOKEN="your_token"
```

### 3. 使用

```bash
# 分析论文，生成深度文章
请帮我分析这篇论文：/path/to/paper.pdf

# 生成论文漫画
请把这篇论文做成漫画：/path/to/paper.pdf
```

---

## paper-analyzer

将学术论文转化为深度技术文章，支持 3 种写作风格。

### 效果展示

<table>
<tr>
<td align="center" width="33%">
<b>academic</b><br/>学术型（默认）<br/><br/>
正式严谨，适合学术分享
</td>
<td align="center" width="33%">
<b>storytelling</b><br/>故事型<br/><br/>
生动比喻，适合公众号
</td>
<td align="center" width="33%">
<b>concise</b><br/>精炼型<br/><br/>
表格列表，快速阅读
</td>
</tr>
</table>

![风格对比](images/styles_comparison.png)

### 可选功能

| 功能 | 说明 | 效果 |
|------|------|------|
| **公式讲解** | 插入公式图片，详解符号含义 | ![](images/formula_feature.png) |
| **代码分析** | 论文概念与 GitHub 源码对照 | ![](images/code_feature.png) |

---

## paper-comic

将学术论文转化为 10 页教育漫画，4 种画风可选。

### 画风选择

<table>
<tr>
<td align="center" width="25%"><img src="examples/paper-comic/cover-classic.png" width="150"/><br/><b>classic</b><br/>通用（默认）</td>
<td align="center" width="25%"><img src="examples/paper-comic/cover-tech.png" width="150"/><br/><b>tech</b><br/>AI/计算机</td>
<td align="center" width="25%"><img src="examples/paper-comic/cover-warm.png" width="150"/><br/><b>warm</b><br/>心理学/教育</td>
<td align="center" width="25%"><img src="examples/paper-comic/cover-chalk.png" width="150"/><br/><b>chalk</b><br/>数学/物理</td>
</tr>
</table>

### 完整示例：Engram 论文

<table>
<tr>
<td align="center"><img src="examples/paper-comic/00-cover.png" width="100"/><br/>封面</td>
<td align="center"><img src="examples/paper-comic/01-page.png" width="100"/><br/>1</td>
<td align="center"><img src="examples/paper-comic/02-page.png" width="100"/><br/>2</td>
<td align="center"><img src="examples/paper-comic/03-page.png" width="100"/><br/>3</td>
<td align="center"><img src="examples/paper-comic/04-page.png" width="100"/><br/>4</td>
<td align="center"><img src="examples/paper-comic/05-page.png" width="100"/><br/>5</td>
</tr>
<tr>
<td align="center"><img src="examples/paper-comic/06-page.png" width="100"/><br/>6</td>
<td align="center"><img src="examples/paper-comic/07-page.png" width="100"/><br/>7</td>
<td align="center"><img src="examples/paper-comic/08-page.png" width="100"/><br/>8</td>
<td align="center"><img src="examples/paper-comic/09-page.png" width="100"/><br/>9</td>
<td align="center"><img src="examples/paper-comic/10-page.png" width="100"/><br/>10</td>
<td></td>
</tr>
</table>

---

## 兼容性

支持以下 AI 编程助手：

- Claude Code
- Cursor
- Codex
- Windsurf
- 其他支持 Claude Code 技能的工具

---

## 致谢

- **baoyu-gemini-web** - 图片生成后端，基于 [JimLiu/baoyu-skills](https://github.com/JimLiu/baoyu-skills)
- **MinerU** - PDF 高精度解析，来自 [mineru.net](https://mineru.net)

---

## License

MIT
