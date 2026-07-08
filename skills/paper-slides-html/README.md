# paper-slides-html — 浏览器幻灯片

**把论文变成纯 HTML/CSS 幻灯片，零依赖、可编辑、可直接分享。**

## 快速开始

```bash
/paper-slides-html https://arxiv.org/abs/1706.03762 --theme academic
/paper-slides-html paper.pdf --theme dark --slides 12
```

## 三种主题

| 主题 | 风格 | 适合 |
|------|------|------|
| **academic** | 学术蓝白风 | 论文组会、答辩 |
| **minimal** | 极简黑白风 | 技术分享 |
| **dark** | 深色科技风 | 公开演示 |

## 对比 paper-deck

| | paper-slides-html | paper-deck |
|---|---|---|
| 依赖 | 无 | 需要生图后端 |
| 可编辑 | 直接改 HTML | 改 prompt 重生成 |
| 大小 | ~50KB | ~50MB |
| 出稿速度 | 秒级 | 分钟级（生图耗时） |
| 视觉效果 | CSS 设计 | AIGC 无限风格 |
