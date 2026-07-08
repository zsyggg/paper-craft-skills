# Prompt: Architecture Comparison — How Each Method Injects Position

Create one complete 16:9 comparison slide.

Purpose:
Show how each method injects position information into the attention mechanism differently.

Papers:
- Sinusoidal (blue): "PE(pos,2i)=sin(pos/10000^(2i/d))" — added to input embedding BEFORE attention
- REL (orange): "R_{i-j} bias table" — added to attention score DURING attention
- RoPE (green): "q·k rotated by mθ" — applied to Q/K vectors BEFORE attention, then standard dot product
- ALiBi (purple): "-|i-j|×m linear penalty" — added to attention score AFTER query-key dot product

Style preset: journal-minimal

Composition:
Horizontal timeline/ladder showing:
| Step: Input → Before Attention → During Attention → After Attention |
| Sinusoidal: [add PE here] — — — — — — — — — — |
| REL: — — — — — [add bias here] — — — — — |
| RoPE: — — [rotate Q/K] — standard dot product — |
| ALiBi: — — — — — — — — — — [add linear bias] |

Highlight the "when" box with each method's color.
Each row = one method, same timeline axis.

Constraints:
- Clearly show at which stage position is injected
- Use the same attention formula as reference backbone
- Highlight the difference with colored boxes
