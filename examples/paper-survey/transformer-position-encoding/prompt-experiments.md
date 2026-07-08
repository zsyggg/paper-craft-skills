# Prompt: Experimental Results Comparison

Create one complete 16:9 comparison slide.

Purpose:
Compare the 4 position encoding methods on extrapolation length and perplexity.

Papers (same colors):
- Sinusoidal (blue): "WikiText-103 ppl 26.0 | Extrapolation 1x: ✓ | 2x: ✗ | 5x: ✗"
- REL (orange): "WikiText-103 ppl 24.0 | Extrapolation 1x: ✓ | 2x: △ | 5x: ✗"
- RoPE (green): "WikiText-103 ppl 23.5 | Extrapolation 1x: ✓ | 2x: ✓ | 5x: △"
- ALiBi (purple): "WikiText-103 ppl 23.2 | Extrapolation 1x: ✓ | 2x: ✓ | 5x: ✓"

Style preset: journal-minimal

Composition:
- Top: title "Extrapolation Benchmark: Position Encoding Methods"
- Main visual: grouped bar chart showing perplexity
  - X-axis: "1x length" | "2x length" | "5x length"
  - Y-axis: "Perplexity ↓"
  - 4 colored bars per group (blue, orange, green, purple)
  - Sinusoidal bar at 5x = missing (greyed out)

- Bottom: annotation table
  | Method | Params | Extrapolation | Best for |
  | Sinusoidal | 0 | 1x only | Standard transformers |
  | REL | O(L²) | 2x | Segment-level processing |
  | RoPE | 0 | 5x △ | LLMs (LLaMA, etc.) |
  | ALiBi | 0 | 5x ✓ | Long-context inference |

Constraints:
- Simple clean bar chart, no 3D
- Data labels on each bar
- Sinusoidal at 5x = "N/A" with grey
- No decorative chart junk
