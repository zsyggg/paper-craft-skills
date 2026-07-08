# Paper-Survey Example: Transformer Position Encoding Comparison

This directory demonstrates paper-survey's **comparison-figures** mode output for 4 position encoding methods.

## Comparison Scope

| Method | Paper | Year | Core Idea |
|--------|-------|------|-----------|
| Sinusoidal | Attention Is All You Need | 2017 | Fixed sine/cosine functions |
| Relative Position (REL) | Transformer-XL | 2019 | Bias table for relative offsets |
| Rotary (RoPE) | RoFormer | 2021 | Rotate Q/K by position angle |
| ALiBi | Train Short, Test Long | 2021 | Linear bias to attention scores |

## Output Prompts

| File | Description |
|------|-------------|
| prompt-overview.md | Comparison overview figure prompt |
| prompt-architecture.md | Method path comparison prompt |
| prompt-experiments.md | Experimental results comparison prompt |
| prompt-timeline.md | Research evolution timeline prompt |

## Quick Start

```bash
/paper-survey --topic "Transformer position encoding" \
  --papers paper1.pdf paper2.pdf paper3.pdf paper4.pdf \
  --format comparison-figures
```
