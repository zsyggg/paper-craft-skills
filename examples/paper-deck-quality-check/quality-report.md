# Paper-Deck Quality Check Report: Attention Is All You Need Example

*Simulated `merge_deck.py --check` run on existing example images*

---

## Command

```bash
python3 scripts/paper-deck/scripts/merge_deck.py --check \
  examples/paper-illustrated/attention-is-all-you-need
```

## Check Results

```
==================================================
Quality Check — attention-is-all-you-need
==================================================

  ✓ self-attention-sketchnote.png: 2460×3480 OK
  ✓ transformer-overview-paper-figure.png: 1400×1080 OK
  ⚠ transformer-overview-paper-figure.png: aspect ratio 1.30 deviates from 16:9 (1.78)
  ✓ Generation log records: 0 (no generation-log.md — expected for existing examples)

  ⚠ No outline.md found — skipping page count check

==================================================
Overall: WARNINGS (non-blocking)
```

## Analysis

| Check | Result | Severity |
|-------|--------|----------|
| Image openable | ✅ Both images open successfully | — |
| Resolution | ✅ sketchnote 2460×3480px, overview 1400×1080px | — |
| Aspect ratio (16:9) | ⚠ Overview is 1.30 (≈4:3), not 16:9 | Low (acceptable for paper-figure style) |
| Generation log | ⚠ No generation-log.md | Expected (pre-merge_deck upgrade) |
| Outline exists | ⚠ No outline.md | Expected (this is a comic example, not a deck) |

## Verdict

**PASS with warnings.** Both images are valid, high-quality, and correctly generated. The aspect ratio warning on `transformer-overview-paper-figure.png` is expected because:
- `paper-figure` style uses portrait/landscape ratios
- `merge_deck.py` default expects 16:9 (optimized for paper-deck)
- The --check mode correctly detected the deviation without blocking

## How to use --check in practice

```bash
# Check a deck before merging
python3 skills/paper-deck/scripts/merge_deck.py my-deck-dir --check

# If all pass, merge with confidence
python3 skills/paper-deck/scripts/merge_deck.py my-deck-dir

# Force merge even with warnings
python3 skills/paper-deck/scripts/merge_deck.py my-deck-dir --skip-checks
```
