# paper-tldr — 论文极致摘要

**10 秒内理解一篇论文的核心。三种粒度：一句话 / 一段话 / 三个要点。**

## 快速开始

```bash
/paper-tldr https://arxiv.org/abs/1706.03762
/paper-tldr paper.pdf --level all --language Chinese
```

## 三层摘要

| 粒度 | 长度 | 适合 |
|------|------|------|
| **one-liner** | ≤ 30 字 | 首页、推文 |
| **TL;DR** | ≤ 150 字 | 群聊、摘要 |
| **bullet** | 3 × ≤ 20 字 | 笔记、卡片 |

## 示例

> **One-liner:** 用纯注意力替代循环做序列转录，翻译 SOTA 且训练更快。
>
> **TL;DR:** RNN 顺序处理导致训练慢。Transformer 用自注意力机制并行建模全局依赖。在 WMT 翻译上达到 BLEU 28.4 SOTA，训练仅需 3.5 天。
>
> **Bullet:**
> 1. 自注意力替代循环实现并行
> 2. 多头注意力学习多子空间
> 3. 位置编码无递归表达顺序
