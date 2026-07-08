# paper-summary 示例：Attention Is All You Need

本文档展示了 `paper-summary` 技能 infographic 模式的完整工作流程产出。

## 快速开始

```bash
/paper-summary https://arxiv.org/abs/1706.03762 --mode infographic --language English
```

## 产出文件

| 文件 | 说明 |
|------|------|
| `summary-brief.md` | Step 1-2：分析提取 + 确认记录 |
| `prompt-infographic.md` | Step 3-4：最终可执行的生图 prompt |
| `expected-output.md` | 预期输出描述和布局说明 |

## 工作流复盘

```text
Step 1: 读论文 → 提取核心信息
  ↓
Step 2: 确认 → infographic, English, 方法驱动
  ↓
Step 3: 写 prompt → 按 infographic prompt 模板填写
  ↓
Step 4: 生成 → 喂入生图后端 → images/summary-infographic.png
  ↓
Step 5: 使用 → 分享到 Twitter/WeChat/公众号
```

> 🎯 将 `prompt-infographic.md` 复制到支持 image generation 的工具即可生成实际图片。
