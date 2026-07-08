# paper-summary — 论文视觉摘要

**把论文变成一页 AIGC 视觉摘要/信息图，一眼看懂核心内容。**

## 核心能力

| 模式 | 输出 | 适合 |
|------|------|------|
| **infographic** | AIGC 信息图 | 社交媒体分享、公众号封面、笔记 |
| **card-summary** | 卡片式摘要 | 论文速览、文献管理卡片 |
| **timeline-figure** | 研究演进图 | 方法改进/实验流程时间线 |

## 快速开始

```bash
/paper-summary https://arxiv.org/abs/1706.03762
/paper-summary /path/to/paper.pdf --mode infographic --language Chinese
/paper-summary notes.md --mode card-summary
```

## 三种模式

| 模式 | 布局 | 信息密度 | 适合场景 |
|------|------|---------|----------|
| **infographic** | 竖版长图 | 高 | 公众号、小红书、Twitter |
| **card-summary** | 1:1 方卡 | 中 | Zotero、Notion、文献卡片 |
| **timeline-figure** | 横版 16:9 | 中 | README 首图、PPT 插入 |
