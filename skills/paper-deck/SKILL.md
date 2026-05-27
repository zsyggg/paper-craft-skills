---
name: paper-deck
description: |
  将论文、技术文章或知识内容制作成高真实感的 AIGC 幻灯片。先做叙事结构和逐页视觉导演，再调用生图模型生成每一页 16:9 slide image，最后合成为 PPTX/PDF。适合论文汇报、组会、公开课、技术分享、商业化研究展示；当用户提到“论文PPT”“AI生成PPT”“不像AI的PPT”“高质感幻灯片”“逐页生图PPT”时使用。
---

# Paper Deck — Visual Slide Director

把论文/知识内容做成**看起来真的被设计过**的幻灯片。

核心路线不是用 PPT 对象硬摆版式，而是：

1. 先理解内容，做出 deck brief 和逐页叙事。
2. 为每一页写清楚“这页要让观众看到什么、感到什么、记住什么”。
3. 用生图模型生成 16:9 slide image。
4. 合成 PPTX/PDF，并保留 prompts 作为可返修的源文件。

## 不可绕过的生图要求

Paper Deck 的 V1 是 **raster-first AIGC slide image** 工作流。除非用户明确要求“不要生图”“用代码画图”“只要可编辑 PPT”或“使用 HTML/SVG/Canvas 生成”，否则必须调用真实的 raster image generation backend 为每一页生成图片。

严格禁止把以下产物冒充为本 skill 的“生图页”：

- 用 Python/Pillow、SVG、HTML/CSS、Canvas、Mermaid、matplotlib、PPT shapes 或任何本地绘图代码直接画出的整页图片
- 用模板、纯排版脚本、截图、占位图或手工组合元素替代生图后端输出
- 先本地画整页，再仅做轻微滤镜/后处理后当作 AIGC slide image

允许的本地处理仅限：

- 移动、复制、重命名生图后端输出文件
- 必要的格式转换、压缩、尺寸校验、PPTX/PDF 合成
- 用户明确选择混合方案时，在生图背景上叠加少量可编辑文字层；此时必须在 `deck-brief.md` 和交付说明中明确记录“混合文字层”，不能声称整页文字都由生图模型完成

如果当前环境没有可用的 raster image generation backend，必须停止并说明缺少生图后端；不要退化成本地绘图替代方案。

## 何时使用

适合：
- 论文组会、答辩、reading group、技术分享
- 需要“一眼不像模板 PPT”的视觉汇报
- 用户愿意接受每页是高质感图片，优先追求整体观感和传播效果
- 需要逐页返修：重做第 N 页、换风格、加真实感、减少 AI 味

不适合：
- 需要多人在 PowerPoint 里精细编辑每个文本框
- 大量表格、财务报表、合规材料
- 需要准确复制已有企业 PPT 母版

如果用户需要完全可编辑的 PPT，说明本 skill 的 V1 是 raster-first；可改用常规 PPTX 工具，或生成“图片背景 + 可编辑文字层”的混合方案。

## 工作流

### Step 1: 输入分析

接受：
- arXiv / DOI / 网页链接
- PDF 路径
- Markdown / 文本 / 文章
- 已有大纲
- 参考图片或参考 PPT 截图

如果是论文，优先复用 `paper-analyzer` 的阅读方式：读摘要、方法、实验、图表、结论；必要时搜索代码仓库。目标不是写长文，而是提取适合做 slide 的核心叙事。

输出并保存 `analysis.md`：
- 主题、受众、汇报场景
- 论文/内容的 1 句话主张
- 3-5 个必须讲清楚的核心点
- 推荐页数、推荐风格、语言
- 需要生成的图像类型：封面、机制图、流程图、数据页、结论页等
- 可直接使用的真实素材：论文 Figure/Table、PDF 截图、用户提供的截图、代码截图、实验曲线

### Step 2: 生成前确认

默认必须确认，不要直接生成图片。除非用户明确说“直接生成/不用确认/按默认来”。

询问时控制在 3 个问题以内：

1. 页数和用途：组会 / 答辩 / 公开分享 / 商业汇报，需要几页？
2. 风格：见 `references/style-system.md`。
3. 是否插入真实素材：是否允许从 PDF/论文图表中截图，或由用户提供截图/图片？如果允许，说明预计第几页使用哪些真实素材。

推荐话术：

```text
我建议做 12 页，风格用 journal-minimal：像 Nature/IEEE 论文图 + 正式学术汇报，清晰、克制、不花哨。
也可以换成 business-research 做商业研究分享，warm-notes 做手记风，或 liquid-glass 做 Apple 式玻璃质感。
这篇论文我建议在第 4 页插入原论文方法图局部截图，第 8 页插入实验曲线/表格截图，再基于这些真实素材做设计化排版。
确认后我会先生成 outline.md 和每页 prompt，再逐页出图并合成 PPTX/PDF。
```

### Step 3: Deck Brief

保存 `deck-brief.md`。必须包含：

- `style_preset`
- `audience`
- `slide_count`
- `language`
- `visual_rules`
- `do_not_use`
- `reference_images`（如有）
- `source_visual_plan`：哪些页使用真实图表/截图，来源和处理方式

风格细节按需读取 `references/style-system.md`。
真实素材策略按需读取 `references/source-visuals.md`。

### Step 4: Outline

保存 `outline.md`。每页用固定结构：

```markdown
## 01. Slide Title
- Role: cover / context / method / mechanism / evidence / result / takeaway
- Message: 这一页唯一要讲清楚的观点
- Visual: 画面主视觉和构图
- Text: 页面上允许出现的短文字
- Evidence: 引用的论文图表/公式/实验数据/代码位置
- Source visual: 是否使用真实截图/论文图表；来源、裁剪范围和落位
- Repair handle: 后续返修时可引用的定位描述
```

规则：
- 每页只承载一个主观点。
- 页面文字尽量少；复杂解释放 speaker script 或备注里。
- 机制页优先画“输入 → 处理 → 输出”，不要画抽象灵感。
- 数据页只放最有说服力的 1-3 个数字。
- 真实论文图/截图通常比凭空生成更可信；能用真实素材时优先规划真实素材落位。
- 不要过度留白。主视觉、图表或证据区域通常应占画面 60%-80%，除非是封面或章节页。
- 8 页以上必须有节奏变化：封面、问题、方法、机制、证据、结论交替。

### Step 5: Prompt Files

每页必须先写 prompt 文件，再调用任何生图工具。

路径：

```text
paper-deck/{topic-slug}/
├── analysis.md
├── deck-brief.md
├── outline.md
├── prompts/
│   ├── 01-slide-cover.md
│   ├── 02-slide-context.md
│   └── ...
├── images/
│   ├── 01-slide-cover.png
│   ├── 02-slide-context.png
│   └── ...
├── {topic-slug}.pptx
└── {topic-slug}.pdf
```

Prompt 写法读取 `references/prompt-template.md`。

硬规则：
- prompt 必须明确 16:9。
- prompt 里要写清楚风格、构图、文字语言、文字数量限制。
- 不要让模型生成页码、logo、水印、PPT 外壳。
- 如果需要精准文字，尽量减少图片内文字；可以后续做混合文字层。
- 如果本页使用真实素材，prompt 必须说明素材如何作为画面的一部分：嵌入、裁切、玻璃面板承载、旁注、放大框，而不是让模型凭空重画事实。

### Step 6: 生成图片

图片后端选择：

1. Codex 环境优先用内置 `imagegen`。
2. 如果用户指定 `baoyu-imagine`、Gemini、OpenAI、Seedream 等后端，按用户指定。
3. 如果没有可用生图后端，停止并告诉用户需要一个 raster image backend。

生图门禁：
- 在调用任何生图工具之前，必须已经写好对应页的 `prompts/NN-*.md`。
- 每一页最终进入 `images/` 的主图必须来自真实 raster image generation backend。
- 不允许用 Python/Pillow、SVG、HTML/CSS、Canvas、Mermaid、matplotlib、PPT shapes 或本地绘图脚本生成整页主图来替代生图。
- 不允许因为担心中文文字错误，就绕过生图后端改成本地绘制整页。正确做法是减少图片内文字、改 prompt 重生成，或在用户同意的情况下使用“生图背景 + 可编辑文字层”的混合方案。
- 如果使用混合文字层，`images/` 中仍必须保留每页的生图背景或生图整页来源，并在 `deck-brief.md` 记录哪些文字是后叠加的。
- 生成后要在 `generation-log.md` 记录每页使用的后端、prompt 文件、输出文件、生成时间；没有生成记录的图片不能作为最终交付页。

生成策略：
- 先生成第 1 页作为风格锚点。
- 后续页如果后端支持 reference image，就用第 1 页作为风格参考，降低漂移。
- 每 3-4 页检查一次缩略图，发现风格漂移就修 prompt 再继续。
- 保存失败页，不要覆盖成功页。

### Step 7: 合成 PPTX/PDF

生成完图片后运行：

```bash
python3 <SKILL_ROOT>/scripts/merge_deck.py paper-deck/{topic-slug}
```

脚本会读取 `images/NN-*.png|jpg|webp`，输出同名 `.pptx` 和 `.pdf`。每张图片铺满一页 16:9。

### Step 8: 质量检查

交付前按 `references/quality-gate.md` 检查：

- 是否一眼像真实设计作品，而不是模板堆砌
- 每页是否只有一个主观点
- 是否有过多无意义留白；关键内容是否占据足够画面
- 真实素材页是否明确记录来源、页码/图号和落位
- 风格是否一致
- 图片文字是否清晰、无错别字、无伪字
- 是否存在 AI 常见问题：假 UI、假 logo、乱码标签、过度赛博、塑料 3D、无意义装饰
- `generation-log.md` 是否存在，且每一页都记录了真实 raster image generation backend、prompt 文件和输出文件
- 是否存在本地绘图/模板/截图冒充生图页；如有，必须重做或明确改成用户确认过的非 paper-deck 路线
- PPTX/PDF 是否能打开，页数是否正确

### Step 9: 返修

返修时永远先改源文件：

| 用户说 | 操作 |
|---|---|
| “第 5 页更学术一点” | 改 `prompts/05-*.md`，保留旧图，生成新图 |
| “统一成第 1 页的质感” | 把第 1 页风格锚点追加到相关 prompts |
| “第 7 页文字太多” | 修改 outline 的 Text，再改 prompt |
| “只重做背景，不动内容” | 在 prompt 中保留 Message/Text，重写 Visual |
| “新增一页机制细节” | 更新 outline，新增 prompt，生成图片，重跑合成脚本 |

不要用程序在生成图上涂改文字。文字错了就改 prompt 重生成，或切换到混合文字层方案。

## 参考文件

- `references/style-system.md`：风格预设和选择规则
- `references/layouts.md`：常用页面角色与构图
- `references/source-visuals.md`：PDF 截图、论文图表、用户图片的使用策略
- `references/prompt-template.md`：逐页生图 prompt 模板
- `references/quality-gate.md`：交付前检查和返修标准
