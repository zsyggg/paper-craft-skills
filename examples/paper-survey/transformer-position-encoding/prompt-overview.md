# Prompt: Comparison Overview — 4 Position Encoding Methods

Create one complete 16:9 presentation slide for a multi-paper comparison.

Purpose:
Compare 4 position encoding methods at a glance — their core ideas, formulas, and key properties.

Papers:
- Sinusoidal (blue #2563eb): "Fixed sine/cosine, add to input, 0 params"
- REL / Transformer-XL (orange #ea580c): "Relative bias table, segment-level, O(L²) params"
- RoPE / RoFormer (green #16a34a): "Rotate Q/K by angle, zero params, relative=absolute"
- ALiBi (purple #7c3aed): "Linear bias by distance, zero params, best extrapolation"

Style preset: journal-minimal
Comparison layout: 4-column grid, one column per method

Composition:
- Top: title "Transformer Position Encoding: 4 Approaches Compared"
- 4 equal columns, each with:
  - Method name (colored header bar)
  - Core formula (simplified, 1 line)
  - 3 key property tags
  - "Free parameter" / "No params" label
  - Extrapolation: checkmark or cross
- Bottom: shared annotation "All four encode position information into attention without RNNs"

Color Rules:
- Sinusoidal: blue (#2563eb)
- REL/Transformer-XL: orange (#ea580c)
- RoPE: green (#16a34a)
- ALiBi: purple (#7c3aed)
- Common elements: neutral gray (#6b7280)

Constraints:
- All text readable at 16:9 slide size
- Each method column visually distinct
- Same shape language across all methods
- No fake formulas, no decorative elements
