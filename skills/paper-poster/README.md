# paper-poster — 学术海报生成

**把论文变成高质感 AIGC 学术海报（会议展板/答辩海报/课题展示）。**

## 核心能力

| 输入 | 输出 |
|------|------|
| 论文URL/PDF | 学术海报图片（PNG） |
| 已有方法图 + 文字 | 海报 PDF（可打印） |
| 多篇论文/一个课题 | 课题总览海报 |

## 快速开始

```bash
/paper-poster https://arxiv.org/abs/1706.03762
/paper-poster /path/to/paper.pdf --style conference-wide --orientation portrait
/paper-poster --topic "我的课题" --papers paper1.pdf paper2.pdf
```

## 海报风格

| 风格 | 适合 |
|------|------|
| **conference-wide** | 会议宽幅海报（最多内容，标准学术风） |
| **defense-poster** | 答辩海报（清晰结构，导师友好） |
| **research-showcase** | 课题展示（高视觉冲击，传播感） |

## 输出规格

- 宽幅：48×36 英寸（conference-wide）
- 竖版：36×48 英寸（defense-poster）
- 可打印 PDF + 屏幕分享 PNG
