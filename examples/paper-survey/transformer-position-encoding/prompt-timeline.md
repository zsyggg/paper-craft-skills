# Prompt: Research Evolution Timeline

Create one complete 16:9 timeline slide.

Purpose:
Show the evolution of position encoding methods from 2017 to 2021, highlighting each method's innovation.

Content (chronological):

2017 — Sinusoidal (blue)
"Vaswani et al. · Attention Is All You Need"
Milestone: First transformer, fixed position encoding
↓ "Limitation: cannot extrapolate beyond trained length"

2019 — Relative Position (orange)
"Dai et al. · Transformer-XL"
Milestone: Relative bias enables segment-level recurrence
↓ "Limitation: O(L²) parameter cost"

2021 — RoPE (green)
"Su et al. · RoFormer"
Milestone: Rotation = relative position from absolute, zero params
↓ "Adopted by LLaMA, Mistral, modern LLMs"

2021 — ALiBi (purple)
"Press et al. · Train Short, Test Long"
Milestone: Simple linear bias, best extrapolation
↓ "Ideal for long-context inference"

Bottom annotation:
"The evolution shows a clear trend: simpler formulas, zero parameters, better extrapolation"

Style preset: journal-minimal

Composition:
- Horizontal timeline arrow from left to right (2017 → 2021)
- Each method as a card above/below the timeline
- Cards connected to timeline with dots
- Each card: year + method name + color header + author + one-line milestone
- Arrows between cards showing "inspired by" / "improves upon" relationships
- Bottom: trend annotation

Constraints:
- Clean horizontal timeline
- 4 cards, evenly spaced
- Same card size for all four
- Trend annotation at bottom with subtle highlight
