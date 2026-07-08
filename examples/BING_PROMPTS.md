# 🎨 paper-craft-skills — 6 Prompts for Bing Image Creator

把下面每个 prompt 复制到 **https://bing.com/create**（登录微软账号即可免费生成）。

---

## 1️⃣ Paper Summary — 信息图

**复制以下内容到 Bing：**

```
Create a vertical 2:3 infographic summarizing this research paper. Style: modern infographic, clean white background, restrained color palette. Language: English.

Content blocks:
1. TITLE AREA: "Attention Is All You Need" (bold, dark navy #1e3a5f) - "Transformer — Sequence Transduction Without RNNs" - Vaswani et al. · NeurIPS 2017
2. PROBLEM: "RNNs process tokens sequentially → slow training, long paths"
3. METHOD OVERVIEW: Show simplified Transformer architecture horizontally with [Input Embedding] → [Self-Attention] → [Feed-Forward] → [Output], highlight Self-Attention box in blue
4. KEY RESULTS: Three data callouts — "28.4 BLEU Translation SOTA" | "3.5 days train 8 GPU" | "O(1) path Any→Any"
5. CONTRIBUTIONS: Self-Attention, Multi-Head, Position Encoding

Clean white background, blue accent #2563eb, sans-serif typography, no fake data.
```

---

## 2️⃣ Conference Poster — 学术海报

**复制以下内容到 Bing：**

```
Academic conference poster 48x36 inches landscape. Title "Attention Is All You Need" - Vaswani et al. NeurIPS 2017. Left column: problem & motivation, method pipeline. Center-right main area (largest element): full Transformer architecture diagram with Encoder stack (Input+PosEnc→Multi-Head Self-Attention→Add&Norm→FFN→Add&Norm ×6) and Decoder stack (Masked Self-Attention→Cross-Attention→FFN ×6) → Linear → Softmax. Highlight attention modules in blue. Bottom left: three data cards "BLEU 28.4 SOTA" "3.5 days 8 GPU" "O(1) paths". Bottom bar: conclusion + 4 references. White background, navy #1e3a5f titles, blue #2563eb accent, clean vector style, no fake data, no logos, no watermarks.
```

---

## 3️⃣ Survey — Overview Comparison

**复制以下内容到 Bing：**

```
16:9 comparison slide comparing 4 position encoding methods for Transformers. Four-column grid: 1) Sinusoidal (blue) "Fixed sine/cosine, add to input, 0 params" 2) REL/Transformer-XL (orange) "Relative bias table, O(L²) params" 3) RoPE (green) "Rotate Q/K by angle, zero params" 4) ALiBi (purple) "Linear bias by distance, zero params, best extrapolation". Each column has method name, simplified formula, property tags, extrapolation checkmark. Top title "Transformer Position Encoding: 4 Approaches Compared". Journal-minimal style, clean white background, consistent card shapes, readable labels.
```

---

## 4️⃣ Survey — Architecture Comparison

**复制以下内容到 Bing：**

```
16:9 comparison slide showing WHEN each position encoding method injects position into attention. Horizontal timeline: Input → Before Attention → During Attention → After Attention. Four rows: 1) Sinusoidal (blue): "add PE here" at Input stage 2) REL (orange): "add bias here" at During Attention 3) RoPE (green): "rotate Q/K" at Before Attention then standard dot product 4) ALiBi (purple): "add linear bias" at After Attention. Each method's injection point highlighted with colored box. Title "Position Injection: Where Each Method Acts". Journal-minimal style, clean white background, consistent timeline axis.
```

---

## 5️⃣ Survey — Experiments Comparison

**复制以下内容到 Bing：**

```
16:9 comparison slide with grouped bar chart comparing extrapolation performance of 4 position encoding methods. X-axis: "1x length" | "2x length" | "5x length". Y-axis: "Perplexity ↓". Four colored bars per group: Sinusoidal (blue), REL (orange), RoPE (green), ALiBi (purple). Sinusoidal missing at 5x shown as grey N/A. Bottom annotation table with columns: Method, Params, Extrapolation, Best for. Data: Sinusoidal 0 params 1x only, REL O(L²) 2x, RoPE 0 5x, ALiBi 0 5x. Title "Extrapolation Benchmark". Journal-minimal style, clean chart, data labels on bars.
```

---

## 6️⃣ Survey — Evolution Timeline

**复制以下内容到 Bing：**

```
16:9 horizontal timeline showing evolution of position encoding methods 2017→2021. Four cards along timeline: 2017 Sinusoidal (blue) "Vaswani et al. First transformer, fixed PE" connected to 2019 REL (orange) "Dai et al. Relative bias, segment-level" connected to 2021 RoPE (green) "Su et al. Rotation = relative, zero params" connected to 2021 ALiBi (purple) "Press et al. Linear bias, best extrapolation". Arrows between cards showing improvement direction. Bottom annotation: "Simpler formulas, zero parameters, better extrapolation". Journal-minimal style, clean horizontal timeline, equal card sizes.
```

---

## 怎么用

1. 打开 https://bing.com/create
2. 登录微软账号
3. 把上面任意一个 prompt 复制粘贴进去
4. 点击 "创建"
5. 保存生成的图片到对应目录

> 对应目录：
> - 信息图 → `examples/paper-summary/attention-is-all-you-need/images/`
> - 海报 → `examples/paper-poster/attention-is-all-you-need/images/`
> - 对比图 → `examples/paper-survey/transformer-position-encoding/images/`
