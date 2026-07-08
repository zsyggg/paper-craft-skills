# paper-summary 示例：Attention Is All You Need

本文档展示了 `paper-summary` 技能 infographic 模式的完整工作流程产出。

## 快速开始

```bash
/paper-summary https://arxiv.org/abs/1706.03762 --mode infographic --language English
```

## 产出文件

| 文件 | 说明 |
|------|------|
| `summary-brief.md` | 分析提取结果（Step 1） |
| `prompt-infographic.md` | 最终生图 prompt（Step 3-4） |
| `expected-output.md` | 预期输出描述 |

## 工作流复盘

```
1. 读论文 → 提取核心信息到 summary-brief.md
2. 确认模式 → infographic, English, 方法驱动
3. 写 prompt → 按 infographic prompt 模板填写
4. 生成 → 将 prompt 喂入生图后端 → 输出 images/summary-infographic.png
5. 使用 → 分享到 Twitter/WeChat/公众号
```

> 提示：将 `prompt-infographic.md` 的内容复制到任何支持 image generation 的 AI 工具（如 Midjourney、DALL-E、Flux、Codex imagegen）即可生成实际图片。
