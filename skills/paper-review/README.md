# paper-review — 自动论文评审

**对论文进行结构化、多维度的自动审稿，输出 3 份 reviewer reports + 编辑决定。**

## 快速开始

```bash
/paper-review https://arxiv.org/abs/1706.03762
/paper-review /path/to/paper.pdf
```

## 评审维度

| 维度 | 权重 | 说明 |
|------|------|------|
| 新颖性 | 25% | 与现有工作的差异 |
| 技术正确性 | 25% | 方法是否严谨 |
| 实验完整性 | 20% | 基准选择、消融实验 |
| 写作质量 | 15% | 清晰度、图表质量 |
| Reproducibility | 15% | 代码可用性 |

## 输出

```
reviewer-1-report.md  → 审稿人 1
reviewer-2-report.md  → 审稿人 2
reviewer-3-report.md  → 审稿人 3
meta-review.md        → 综合编辑决定
```
