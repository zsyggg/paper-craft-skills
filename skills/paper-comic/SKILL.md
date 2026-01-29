---
name: paper-comic
description: |
  从学术论文生成教育漫画，用视觉叙事解释核心观点和创新点。
  支持 4 种画风：classic（清线）、tech（科技）、warm（温暖）、chalk（粉笔）。
  使用 baoyu-gemini-web 生成图片。
---

# Paper Comic - 论文漫画生成

将学术论文转化为连贯的教育漫画，用视觉叙事让复杂概念易于理解。

## 使用方法

```bash
/paper-comic /path/to/paper.pdf
/paper-comic /path/to/paper.pdf --style tech
/paper-comic  # 然后粘贴论文内容
```

## 画风选项

| 画风 | 特点 | 适用论文 |
|------|------|----------|
| **classic** | 清线风格，简洁专业，易于理解 | 通用，大多数论文（默认） |
| **tech** | 科技风，电路元素，霓虹光效 | AI/计算机/工程类 |
| **warm** | 温暖风，亲和力强，怀旧感 | 心理学/认知科学/教育类 |
| **chalk** | 粉笔风，黑板效果，学术感 | 数学/物理/理论类 |

## 输出结构

```
[output-dir]/
├── outline.md           # 大纲和分镜
├── characters/
│   ├── characters.md    # 角色定义
│   └── characters.png   # 角色参考图
├── prompts/
│   ├── 00-cover.md      # 封面 prompt
│   └── XX-page.md       # 各页 prompt
├── 00-cover.png         # 封面
└── XX-page.png          # 各页漫画
```

**输出目录**：
- 有源文件：`[source-dir]/comic/`
- 无源文件：`comic-outputs/YYYY-MM-DD/[topic-slug]/`

## 工作流程

### 第一步：分析论文

1. 读取论文内容（PDF 或 Markdown）
2. 提取核心信息：
   - 论文标题、作者
   - 研究背景和动机
   - 核心创新点（1-3 个）
   - 关键方法/算法
   - 主要实验结果
3. 根据论文领域自动推荐画风（或使用用户指定）

### 第二步：设计叙事结构

**四段式结构**（适合 8-12 页漫画）：

| 阶段 | 页数 | 内容 |
|------|------|------|
| **引入** | 1-2 页 | 问题背景，为什么需要这个研究 |
| **探索** | 2-3 页 | 现有方法的局限，引出创新点 |
| **核心** | 3-5 页 | 详细讲解创新方法，用比喻可视化 |
| **总结** | 1-2 页 | 实验结果，意义和展望 |

### 第三步：定义角色

创建 `characters/characters.md`：

**必要角色**：
- **导师**：知识讲解者，睿智亲和
- **学生**：代表读者，提问和学习
- **概念化身**（可选）：将抽象概念拟人化

**角色一致性要求**：
- 导师和学生必须出现在 ≥60% 的页面
- 每页明确标注出场角色
- 角色外观在所有页面保持一致

### 第四步：创建分镜

创建 `outline.md`，包含：
- 元数据（标题、画风、页数）
- 封面设计
- 每页的分格布局和内容

**分镜要求**：
- 每页 3-5 个分格
- 标注每格的角色、场景、对话
- 对话全部使用中文
- 公式用图示表达，不写文字公式

### 第五步：生成图片

使用 baoyu-gemini-web 生成（需要 Google 账号认证）：

```bash
# 获取技能安装路径（假设通过 npx skills add 安装）
SKILL_DIR="$HOME/.claude/skills/baoyu-gemini-web"
# 或者如果在其他位置：
# SKILL_DIR="$HOME/.codex/skills/baoyu-gemini-web"

# 生成角色参考图
npx -y bun "$SKILL_DIR/scripts/main.ts" \
  --promptfiles references/base-prompt.md characters/characters.md \
  --image characters/characters.png \
  --sessionId comic-[topic]-[timestamp]

# 生成各页（使用相同 sessionId 保持一致性）
npx -y bun "$SKILL_DIR/scripts/main.ts" \
  --promptfiles references/base-prompt.md prompts/XX-page.md \
  --image XX-page.png \
  --sessionId comic-[topic]-[timestamp]
```

**关键**：使用相同的 `--sessionId` 确保角色外观一致。

**首次运行**：会打开 Chrome 浏览器进行 Google 账号认证，之后 Cookie 会被缓存。

### 第六步：输出文档

生成 `[topic]-paper-comic.md`：

```markdown
# [论文标题] - 漫画解读

## 概览
- **论文**：[标题]
- **画风**：[选择的画风]
- **页数**：[N]
- **生成时间**：[YYYY-MM-DD]

## 漫画页面

### 封面
![封面](00-cover.png)

### 第 1 页
![第1页](01-page.png)
**内容**：[简述本页讲解的内容]

...

## 核心知识点
1. [知识点1]
2. [知识点2]
3. [知识点3]
```

## 重要原则

### 文字要求
- **所有对话和旁白必须是中文**
- 专业术语：中文 + 英文，如"梯度下降 (Gradient Descent)"
- 文字必须清晰可读，不模糊

### 公式处理
- **不要用文字写公式**
- 用图示/比喻表达公式含义
- 例：梯度下降 → 画小球滚下山坡

### 画面连贯性
- 角色外观全程一致
- 场景风格统一
- 叙事逻辑清晰递进

## 参考文件

- `references/base-prompt.md` - 基础 prompt 模板
- `references/styles/classic.md` - 清线风格
- `references/styles/tech.md` - 科技风格
- `references/styles/warm.md` - 温暖风格
- `references/styles/chalk.md` - 粉笔风格
