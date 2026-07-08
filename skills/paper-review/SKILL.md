---
name: paper-review
description: |
  自动审稿——对论文进行结构化评审，输出三份 reviewer reports + 综合编辑决定。
  评审维度：新颖性、技术正确性、实验完整性、写作质量、 reproducibility。
  当用户提到"审稿""评审""review""peer review""论文评审"时使用。
---

# Paper Review — Automated Peer Review

对论文进行结构化、多维度的自动评审。

## 核心哲学

### 我们的审稿原则

1. **建设性** — 指出问题的同时给出改进建议
2. **结构化** — 多维度评分，不只有"好/不好"
3. **可操作** — 每条评审意见都指向论文中的具体位置
4. **诚实** — 明确区分"我不同意"和"这里错了"

### 我们不做的

- ❌ 只写"好论文"或"差论文"（没有信息量）
- ❌ 无中生有编造问题
- ❌ 对论文没有深入理解就评分

---

## 评审维度

| 维度 | 权重 | 评审内容 |
|------|------|---------|
| Novelty | 25% | 新颖性、与现有工作的差异、边际贡献 |
| Technical Correctness | 25% | 方法是否正确、假设是否合理、证明是否严谨 |
| Experimental Completeness | 20% | 实验设计、基准选择、消融实验、统计显著性 |
| Writing Quality | 15% | 清晰度、组织结构、图表质量、语言 |
| Reproducibility | 15% | 代码可用性、数据公开性、实验细节充分性 |

## 输出

运行后生成：

```text
paper-review/{paper-slug}/
├── review-brief.md          # 论文基本信息
├── reviewer-1-report.md     # 审稿人 1 报告
├── reviewer-2-report.md     # 审稿人 2 报告
├── reviewer-3-report.md     # 审稿人 3 报告
├── meta-review.md           # 综合编辑决定
└── decision.md              # 最终决定和建议
```

## 审稿报告模板

```markdown
# Reviewer Report

Paper: [Title]
Reviewer: [1/2/3]

## Summary (1-2 sentences)

## Score Breakdown

| Dimension | Score (1-10) | Comments |
|-----------|-------------|----------|
| Novelty | X/10 | |
| Technical Correctness | X/10 | |
| Experimental Completeness | X/10 | |
| Writing Quality | X/10 | |
| Reproducibility | X/10 | |
| **Overall** | **X/10** | |

## Strengths
1. ...
2. ...

## Weaknesses
1. ...
2. ...

## Detailed Comments

### Major Issues
1. [Issue with paper section reference]
   - Why it's a problem
   - Suggested fix

### Minor Issues
1. ...

## Questions for Authors
1. ...

## Recommendation
- [ ] Accept
- [ ] Minor Revision
- [ ] Major Revision
- [ ] Reject
```

## 综合决定模板

```markdown
# Meta Review

## Summary of Reviews
- Reviewer 1: Score X/10, Recommendation: [accept/revision/reject]
- Reviewer 2: Score X/10, Recommendation: [accept/revision/reject]
- Reviewer 3: Score X/10, Recommendation: [accept/revision/reject]

## Areas of Agreement
...

## Areas of Disagreement
...

## Editorial Decision
- [ ] Accept
- [ ] Minor Revision
- [ ] Major Revision
- [ ] Reject

## Decision Rationale
...
```

## 参考

- `references/review-criteria.md` — 各维度详细评分标准
- `references/review-examples.md` — 示例 review
