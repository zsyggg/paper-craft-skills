# Expected Output — Infographic Visualization

以下是用 Mermaid 图近似展示最终生成的信息图的视觉布局。

## Layout Mockup

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'primaryColor': '#2563eb', 'lineColor': '#1e3a5f'}}}%%
block-beta
  columns 3

  block:title
    columns 3
    T["Attention Is All You Need"]
    ST["Transformer — Without RNNs"]
    AU["Vaswani et al. · NeurIPS 2017"]
  end

  block:problem
    columns 3
    P["Problem: RNNs process tokens sequentially → slow training"]
  end

  space:3

  block:arch
    columns 3

    block:inp
      I["Input<br/>Embedding"]
    end
    block:pe
      PE["+ Position<br/>Encoding"]
    end
    block:sa
      SA["⚡ Self-<br/>Attention"]
    end
    block:ff
      FF["Feed-<br/>Forward"]
    end
    block:out
      O["Output<br/>Softmax"]
    end

    I --> PE --> SA --> FF --> O
  end
  style SA fill:#2563eb,color:#fff,stroke:#1e3a5f

  space:3

  block:data
    columns 3
    D1["28.4 BLEU<br/>Translation SOTA"]
    D2["3.5 days<br/>8 GPU training"]
    D3["O(1) path<br/>Any→Any attention"]
  end

  block:contrib
    columns 3
    C1["Self-Attention<br/>Global dependencies"]
    C2["Multi-Head<br/>Parallel subspaces"]
    C3["Pos Encoding<br/>Order without RNN"]
  end

  block:footer
    columns 3
    F["arXiv:1706.03762 · Open Source: Tensor2Tensor"]
  end
```

## Visual Description

```
┌─────────────────────────────────────┐
│  Attention Is All You Need          │  ← Title (15%)
│  Transformer — Without RNNs         │
│  Vaswani et al. · NeurIPS 2017      │
├─────────────────────────────────────┤
│  Problem: RNNs sequential → slow    │  ← Problem (10%)
├─────────────────────────────────────┤
│                                     │
│   Input → +Pos → Self-Att → FF → O │  ← Method (35%)
│          ║       ⚡highlight         │     ★核心视觉
│   QKV · Parallel · O(1) paths      │
│                                     │
├──────────┬───────────┬──────────────┤
│ 28.4     │ 3.5 days  │  O(1) path   │  ← Data (25%)
│ BLEU SOTA│ 8 GPU     │  Any→Any     │
├──────────┴───────────┴──────────────┤
│ • Self-Attn  • Multi-Head  • PosEnc │  ← Contrib (10%)
├─────────────────────────────────────┤
│  arXiv:1706.03762                   │  ← Footer (5%)
└─────────────────────────────────────┘
```

## 实际生成效果（文字描述）

- **配色**：纯白底 + 深蓝标题 #1e3a5f + 强调蓝 #2563eb
- **核心视觉**：中间 35% 区域是 Transformer 水平架构简图，Self-Attention 方块用蓝色高亮
- **数据区**：3 个大数字卡片并排，数字 56pt 粗体蓝色
- **字体**：Inter 风格无衬线字体全文统一
- **整体感观**：像 Stripe/Linear 风格的产品信息图，干净、现代、阅读流畅
