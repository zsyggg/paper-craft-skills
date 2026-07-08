# Expected Layout — Conference-Wide Poster

## Layout Map

```
┌──────────────────────────────────────────────────────────────────┐
│  TITLE BANNER (12%)                                               │
│  "Attention Is All You Need" — Vaswani et al. · NeurIPS 2017     │
├─────────────────────────────┬────────────────────────────────────┤
│  PROBLEM & MOTIVATION (18%) │  MAIN METHOD FIGURE (38%)          │
│  • RNNs sequential → slow  │  ┌──────────────────────────────┐  │
│  • Long-range O(n) paths   │  │  Full Transformer            │  │
│  • Prior arch limits       │  │  Encoder-Decoder             │  │
│                             │  │  Self-Attn highlighted blue  │  │
│  PIPELINE (15%)            │  │  (largest visual element)    │  │
│  Input→PosEnc→Attn→FFN→Out│  └──────────────────────────────┘  │
│                             │                                     │
├─────────────────────────────┤  SELF-ATTENTION DETAIL (10%)       │
│  KEY RESULTS (20%)         │  Attention(Q,K,V) =                 │
│  ┌────┬────┬────┐          │  softmax(QK^T/√d)V                  │
│  │28.4│3.5d│O(1)│          │  Multi-head: Concat(h_i)W^O        │
│  │BLUE│days│path│          │                                     │
│  └────┴────┴────┘          │                                     │
├─────────────────────────────┴────────────────────────────────────┤
│  CONCLUSION + REFERENCES (5%)                                    │
│  "Transformer achieves SOTA with pure attention"                 │
│  [1] Vaswani 2017  [2] BERT  [3] GPT-3  [4] ViT                │
└──────────────────────────────────────────────────────────────────┘
```

## Visual Specifications

| Element | Spec |
|---------|------|
| Total size | 48 × 36 in (14400 × 10800 px @ 300dpi) |
| Background | White (#ffffff) |
| Title color | Dark navy (#1e3a5f) |
| Accent color | Blue (#2563eb) |
| Body text | Dark (#1a1a1a) |
| Title font | 96pt bold sans-serif |
| Section headers | 40pt bold |
| Body text | 26pt |
| Data numbers | 56pt bold accent |
| Labels | 20pt |
| Method diagram | Central, 38% of height, vector style |
| Data cards | 3-column grid, numbers largest text on poster after title |
| Output | poster-print.png (14400×10800) + poster.pdf (print-ready) |
