# API Key Guide — Image Generation Backends

Choose any ONE backend below. All can generate our prompt files into real images.

---

## 🥇 Recommended: Replicate (Flux) — FREE $5 credits

**Best balance of quality, speed, and free tier.**

| Detail | Value |
|--------|-------|
| Signup | https://replicate.com/signup |
| Free credits | **$5** (no credit card required initially) |
| Cost per image | ~$0.003 (flux-schnell) ~$0.04 (flux-dev) |
| Images with $5 | **~1,500 (schnell) or ~125 (dev)** |
| Our recommendation | `black-forest-labs/flux-dev` for posters/slides |

**Step-by-step setup:**

```bash
# 1. Sign up at https://replicate.com/signup (GitHub Google account)
# 2. Get your API token at https://replicate.com/account/api-tokens
# 3. Set the token:
export REPLICATE_API_TOKEN="r8_xxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## 🥈 Together AI — FREE $1 credit

**Good quality, simple API.**

| Detail | Value |
|--------|-------|
| Signup | https://together.ai/signup |
| Free credits | **$1** |
| Best model | `black-forest-labs/FLUX.1-dev` |
| Docs | https://docs.together.ai/docs/images-overview |

```bash
# Sign up → get key at https://api.together.ai/settings/api-keys
export TOGETHER_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## 🥉 OpenAI DALL-E 3 — Pay-as-you-go

**Best for text rendering in images. Requires $5+ deposit.**

| Detail | Value |
|--------|-------|
| Signup | https://platform.openai.com/signup |
| Min deposit | **$5** (one-time) |
| Cost per image | $0.04 (1024×1024) $0.08 (1792×1024) |
| Best for | Images with precise text labels |

```bash
# Get key at https://platform.openai.com/api-keys
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxx"
```

---

## 🆓 Totally Free Options (No API Key)

If you can't get any API key, use these:

### 1. Hugging Face Spaces (Free, no key)
- Go to https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev
- Upload prompt → download result
- Manual, but completely free

### 2. Local with Ollama + FLUX
```bash
# Requires GPU + 12GB VRAM
pip install diffusers transformers accelerate torch
python scripts/generate_images.py --backend local
```

---

## 🔄 Generation Workflow

### Option A: Auto (with API key)
```bash
pip install replicate  # or: openai, together
python scripts/generate_images.py --backend replicate --all
# This generates ALL 6 example images automatically
```

### Option B: Manual (no API key)
```bash
python scripts/batch_generate.py --html
# Opens dashboard.html → click any prompt → copy → paste to:
# - midjourney.com
# - https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev
# - bing.com/create (DALL-E 3 free)
```

---

## Cost Estimate (all 6 example images)

| Backend | Total cost | Time |
|---------|-----------|------|
| Replicate flux-schnell | **$0.02** | ~30s |
| Replicate flux-dev | **$0.24** | ~2min |
| Together AI | **$0.20** | ~2min |
| DALL-E 3 | **$0.48** | ~1min |

**All 6 images for less than a cup of coffee.**
