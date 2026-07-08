# paper-survey — 多论文对比综述

**把多篇论文变成对比图解、综述幻灯片和深度对比长文。**

## 核心能力

| 输入 | 输出 |
|------|------|
| 多篇论文URL/PDF | 对比图解（AIGC） |
| 一个研究主题 + 论文列表 | 综述幻灯片（PPTX/PDF） |
| 已有论文阅读笔记 | 深度对比长文（HTML） |

## 快速开始

```bash
/paper-survey https://arxiv.org/abs/1706.03762 https://arxiv.org/abs/1907.04340
/paper-survey --topic "Transformer 系列工作对比" --papers paper1.pdf paper2.pdf paper3.pdf
/paper-survey review.md --format article
```

## 三种输出模式

| 模式 | 输出 | 适合 |
|------|------|------|
| **comparison-figures** | 多张 AIGC 对比图解 | 快速看清多篇论文差异 |
| **survey-deck** | 16:9 综述幻灯片 PPTX/PDF | 组会综述报告 |
| **survey-article** | 深度对比长文 HTML | 文献综述写作 |

## 对比框架

论文对比从以下 5 个维度展开：

1. **问题定义**：解决什么问题？输入输出定义？
2. **方法路径**：核心思路是什么？技术路线差异？
3. **关键设计**：每个设计的动机和效果？
4. **实验表现**：在哪些 benchmark 上有差异？
5. **适用边界**：各自适合什么场景？
