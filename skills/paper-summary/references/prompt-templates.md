# Paper Summary Prompt Templates

## infographic 模式

竖版 2:3，适合分享和传播。

```markdown
Create a vertical 2:3 infographic summarizing this research paper.

Style: modern infographic, clean white background, restrained color palette
Language: {Chinese/English}

Content blocks (from top to bottom):
1. TITLE AREA (15%): Paper title (1 line) + one-line contribution
2. PROBLEM (10%): What problem does this paper solve? (1-2 short phrases)
3. METHOD OVERVIEW (30%): Simple horizontal flow diagram showing input→process→output
4. KEY NUMBERS (25%): 2-3 large data callouts with short labels — the most impressive results
5. CONTRIBUTIONS (15%): 3 bullet points, max 10 characters each
6. PAPER INFO (5%): Authors, year, source

Visual constraints:
- All text readable at screen size
- Use icons sparingly and meaningfully
- Main flowchart must be center-visible
- Do NOT generate fake data or numbers
- Background must be clean white or near-white
- Total text on image: max 200 characters
```

## card-summary 模式

1:1 方卡，适合文献管理卡片。

```markdown
Create a square 1:1 research summary card.

Style: clean minimal card, white background, one accent color
Language: {Chinese/English}

Content (compact, single view):
- Top: Paper title (1 line) + authors + year
- Middle-left: Problem (1 short phrase)
- Middle-center: Method (2-3 lines max)
- Middle-right: Key results (3 numbers with labels)
- Bottom: One-line contribution statement

Visual constraints:
- Extremely compact — think business card format
- All text must be large enough to read on mobile
- No decorative elements, no icons
- Do NOT generate fake data
- Total text: max 120 characters
```

## timeline-figure 模式

16:9 横版，适合 README 或 PPT 插入。

```markdown
Create a horizontal 16:9 timeline/figure summarizing this research method.

Style: clean scientific figure, white/light gray background
Language: {Chinese/English}

Layout:
- Horizontal flow from left to right: 4-6 steps
- Each step: icon/symbol + 1-3 word label
- Arrows connecting steps
- Key innovation step highlighted with accent color
- Bottom: brief annotation for the main contribution

Steps:
1. {step1 name} — {brief description}
2. {step2 name} — {brief description}
...

Visual constraints:
- Clean vector style, no photos or 3D
- One accent color to highlight the innovation
- All labels ≤ 15 characters each
- No fake data
```
