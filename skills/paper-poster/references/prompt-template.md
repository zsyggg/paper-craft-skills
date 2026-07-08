# Poster Prompt Template

## Full Poster Prompt

```markdown
Create one complete academic conference poster image at {width}x{height} inches, 300dpi.

Style preset: {preset}
Orientation: {portrait/landscape}

Title: {paper title}
Authors: {authors}
Affiliation: {affiliation}

Layout structure (from top to bottom):
1. Title banner — full width top, bold title + authors + affiliation
2. Left column (30% width): Problem & Motivation (3-5 bullets)
3. Center-Right (70% width): Method Overview — main visual, largest element
4. Left column bottom: Key Results — 2-3 data callouts
5. Bottom bar full width: Conclusion (1 line + 3 bullet points) + References (4-6 items)

Content to include:
[Detailed content from poster-brief.md]

Visual constraints:
- Use {preset} color palette
- Typography: bold clean sans-serif for titles, readable sans-serif for body
- All text must be legible at arm's length
- Do NOT generate page numbers, logos, watermarks, decorative borders
- Do NOT generate fake data, fake charts, fake numbers
- Maintain clear visual hierarchy — viewer should know where to look first
- When using source visuals, preserve the original content
```

## Panels Reference

| Panel type | Suggested size (% of poster) | Content |
|------------|------------------------------|---------|
| Title banner | 12% height | Title, authors, affiliation |
| Main method figure | 35-40% | Core diagram or architecture |
| Problem panel | 15% | 2-3 bullet points |
| Results panel | 15% | Key numbers, mini charts |
| Conclusion panel | 10% | Takeaway + contributions |
| References bar | 5% | 4-6 references |
