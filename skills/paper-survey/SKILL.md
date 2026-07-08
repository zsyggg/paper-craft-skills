---
name: paper-survey
description: |
  多论文对比综述——输入多篇论文或一个研究主题，自动分析每篇论文的核心方法、关键差异和各自贡献，生成对比图解、综述幻灯片或深度对比长文。
  支持 3 种输出模式：comparison-figures（对比图解）、survey-deck（综述幻灯片）、survey-article（综述长文）。
  当用户提到"对比几篇论文""论文综述""文献综述""这几篇有什么区别""survey""comparison"时使用。
---

# Paper Survey — Multi-Paper Comparison & Survey

把多篇论文/一个研究主题的多项工作组织成有结构的对比综述。

## 核心哲学

### 我们比的不是"好和不好"，而是"各怎么做到的"

一篇论文的对比综述，核心回答三个问题：

1. **各自解决什么问题？** — 问题定义与动机差异
2. **各自怎么做的？** — 方法路径与关键设计差异
3. **各自输在哪里？** — 适用边界与核心 trade-off

### 我们不做的事

- ❌ 只列论文摘要拼在一起（那就不是综述）
- ❌ 无意义的"Paper A 好，Paper B 也好"（没有对比深度）
- ❌ 不读论文就编造对比维度
- ❌ 用文字硬说差异，没有视觉辅助理解

### 每一张对比图的标准

> 一个没读过这些论文的人，看这张对比图能立刻说出每篇论文的核心差异。

能 → 通过。不能 → 重新组织对比维度。

---

## 三种输出模式

### Mode 1: comparison-figures（对比图解）

输出：多张 AIGC 对比图（PNG）

适合：
- 快速向团队展示多篇论文的差异
- 论文阅读笔记中的对比总览
- 技术选型时的方法对比

典型内容：
- **对比总览图**：一张大图并列展示所有论文的核心路径
- **关键维度对比图**：每个对比维度一张图（如架构对比、实验表现对比）
- **演进路线图**：时间线展示该领域的工作演进

### Mode 2: survey-deck（综述幻灯片）

输出：16:9 幻灯片（PPTX + PDF）

复用 paper-deck 的视觉导演工作流，但每页的内容从"一篇论文的机制讲解"变成"多篇论文的对比分析"。

典型页结构：
```text
cover → timeline/overview → problem-definition → 
method-comparison（3-5 pages，每个方法路径一张）→ 
key-design-differences → experimental-comparison → 
trade-off-analysis → summary-recommendation
```

### Mode 3: survey-article（综述长文）

输出：深度对比长文 HTML

复用 paper-analyzer 的写作工作流，但分析对象从一篇论文变为多篇论文的对比。

写作风格同样支持 storytelling / academic / concise。

---

## 工作流

### Step 1: 输入解析

接受：
- 多个 arXiv / DOI / URL
- 多个 PDF 路径
- 一个研究主题 + 自动搜索论文
- 已有论文清单 / Markdown 笔记

如果用户只给了一个研究主题（如"帮我综述一下 Transformer 的位置编码方案"），先搜索该主题的核心论文再继续。

### Step 2: 单篇分析（核心准备）

对每篇论文独立运行 paper-analyzer 的分析流程：

1. 提取标题、摘要、核心创新
2. 理解方法流程（输入→处理→输出）
3. 记录关键设计决策和动机
4. 提取核心实验结果
5. 标注代码状态（有/无/搜索替代实现）

输出到 `papers/` 目录：
```markdown
skills/paper-survey/{survey-slug}/
├── papers/
│   ├── 01-paper-a.md      # 单篇分析
│   ├── 02-paper-b.md
│   └── ...
├── analysis/
│   ├── comparison-matrix.json   # 对比矩阵
│   └── timeline.json            # 时间线
├── prompts/                     # 生图 prompt（Mode 1/2）
├── images/                      # 生成图片（Mode 1/2）
└── output/                      # 最终输出
```

### Step 3: 构建对比矩阵

基于所有单篇分析，自动构建对比矩阵：

```json
{
  "dimensions": [
    {
      "name": "problem_definition",
      "label": "问题定义",
      "papers": {
        "paper-a": "核心问题和输入输出定义",
        "paper-b": "核心问题和输入输出定义"
      }
    },
    {
      "name": "method_core_idea",
      "label": "核心方法思路",
      "papers": {
        "paper-a": "一句话方法描述",
        "paper-b": "一句话方法描述"
      }
    }
  ],
  "timeline": [
    {"year": 2017, "paper": "paper-a", "milestone": "提出该方法"},
    {"year": 2019, "paper": "paper-b", "milestone": "改进xxx"}
  ]
}
```

### Step 4: 确认范围

给用户展示对比矩阵摘要，确认：

1. **输出模式**：comparison-figures / survey-deck / survey-article / all
2. **论文范围**：以上所有论文，还是只看其中几篇？
3. **对比重点**：各维度权重（哪些维度重点展开，哪些轻描淡写）
4. **风格**（仅 survey-deck/survey-article）：参考 paper-deck / paper-analyzer 的风格体系
5. **语言**：中文 / English

默认不直接生成，除非用户明确说"直接生成"。

### Step 5: 生成

#### comparison-figures 模式

1. 为每个对比维度写 prompt（参考 paper-comic 的 prompt 格式）
2. 生成对比总览图 + 关键维度图
3. 输出格式：
   - `images/01-comparison-overview.png`
   - `images/02-dimension-architecture.png`
   - `images/03-dimension-experiments.png`
   - `images/04-evolution-timeline.png`

#### survey-deck 模式

1. 按照对比维度规划 slide 结构
2. 参考 paper-deck 工作流：deck-brief.md → outline.md → prompts/ → images/ → merge
3. 对比页的核心差异用视觉强调（颜色/位置/标注重点标注）

#### survey-article 模式

1. 参考 paper-analyzer 工作流
2. 写作结构围绕对比矩阵展开：
   - 引言：为什么需要对比这些工作
   - 单篇概述：每篇论文 2-3 段
   - 维度对比：每个对比维度一节
   - 综合讨论：各自适用场景和 trade-off
   - 结论和展望

### Step 6: 质量检查

按各模式对应技能的质量标准执行。额外检查：

- [ ] 每篇论文的核心贡献是否准确
- [ ] 对比维度是否一致（不能 Paper A 用维度 X，Paper B 用维度 Y）
- [ ] 差异点是否用视觉/文字明确标注
- [ ] 是否避免了"各说各的"（没有真正对比）
- [ ] 是否有明确的适用场景推荐

---

## 对比视觉设计原则

### 配色策略

- 每篇论文分配一个固定颜色（如 Paper A=蓝、Paper B=橙、Paper C=绿）
- 所有对比图中同一篇论文保持同一颜色
- 强调色用于标注差异点

### 对比布局

| 数量 | 推荐布局 |
|------|---------|
| 2 篇 | 左右并列对比 |
| 3 篇 | 上中下三分 / 左中右三列 |
| 4+ 篇 | 网格布局 + 编号 + 分组色块 |

### 差异高亮

- 同：用相同/相似视觉表示
- 异：用对比色 + 虚线框 + 标注文字强调
- 演进：用时间线 + 箭头表示改进关系

---

## 参考文件

- `references/comparison-framework.md` — 对比维度框架
- `references/survey-outline-template.md` — 各模式大纲模板
- `references/prompt-template-survey.md` — 对比图 prompt 模板
- `scripts/merge_survey.py` — 合成辅助脚本
