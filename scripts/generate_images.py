#!/usr/bin/env python3
"""
paper-craft-skills — Automatic Image Generation Script

Generates all example images using:
  - Replicate (Flux)  → pip install replicate  → export REPLICATE_API_TOKEN
  - OpenAI (DALL-E 3) → pip install openai     → export OPENAI_API_KEY
  - Together AI       → pip install together    → export TOGETHER_API_KEY
  - Local (HuggingFace diffusers) → pip install diffusers torch

Usage:
    # Auto-generate all 6 example images (recommended backend)
    python scripts/generate_images.py --backend replicate --all

    # Generate specific images
    python scripts/generate_images.py --backend replicate summary poster
    python scripts/generate_images.py --backend openai --list

    # Dry-run: show what would be generated
    python scripts/generate_images.py --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent


# ── Image Configurations ────────────────────────────────────────────────────

@dataclass
class ImageJob:
    key: str
    title: str
    prompt_file: str  # relative to ROOT
    output_name: str
    width: int = 1024
    height: int = 1024
    backend_hints: dict = field(default_factory=lambda: {"replicate": "flux-dev", "openai": "dall-e-3"})

    @property
    def prompt_path(self) -> Path:
        return ROOT / self.prompt_file

    def get_prompt_text(self) -> str:
        """Extract the actual prompt from the markdown file."""
        if not self.prompt_path.exists():
            return ""
        with open(self.prompt_path, encoding="utf-8") as f:
            content = f.read()
        # Extract the code block content
        in_block = False
        lines = []
        for line in content.split("\n"):
            if line.strip().startswith("```markdown") or line.strip().startswith("```text"):
                in_block = True
                continue
            if line.strip().startswith("```") and in_block:
                in_block = False
                continue
            if in_block:
                lines.append(line)
        return "\n".join(lines).strip()


JOBS: list[ImageJob] = [
    ImageJob(
        key="summary",
        title="Paper Summary — Infographic",
        prompt_file="examples/paper-summary/attention-is-all-you-need/prompt-infographic.md",
        output_name="examples/paper-summary/attention-is-all-you-need/images/summary-infographic.png",
        width=1080, height=1620,
    ),
    ImageJob(
        key="poster",
        title="Conference Poster — Transformer",
        prompt_file="examples/paper-poster/attention-is-all-you-need/prompt-poster.md",
        output_name="examples/paper-poster/attention-is-all-you-need/images/poster-conference.png",
        width=1440, height=1080,
    ),
    ImageJob(
        key="survey-overview",
        title="Survey — Overview Comparison",
        prompt_file="examples/paper-survey/transformer-position-encoding/prompt-overview.md",
        output_name="examples/paper-survey/transformer-position-encoding/images/survey-overview.png",
        width=1920, height=1080,
    ),
    ImageJob(
        key="survey-architecture",
        title="Survey — Architecture Comparison",
        prompt_file="examples/paper-survey/transformer-position-encoding/prompt-architecture.md",
        output_name="examples/paper-survey/transformer-position-encoding/images/survey-architecture.png",
        width=1920, height=1080,
    ),
    ImageJob(
        key="survey-experiments",
        title="Survey — Experiments Comparison",
        prompt_file="examples/paper-survey/transformer-position-encoding/prompt-experiments.md",
        output_name="examples/paper-survey/transformer-position-encoding/images/survey-experiments.png",
        width=1920, height=1080,
    ),
    ImageJob(
        key="survey-timeline",
        title="Survey — Evolution Timeline",
        prompt_file="examples/paper-survey/transformer-position-encoding/prompt-timeline.md",
        output_name="examples/paper-survey/transformer-position-encoding/images/survey-timeline.png",
        width=1920, height=1080,
    ),
]

JOB_MAP = {j.key: j for j in JOBS}


# ── Backend Implementations ────────────────────────────────────────────────

class BackendError(Exception):
    pass


def _check_env(var: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise BackendError(f"Missing {var}. See API_KEY_GUIDE.md for setup.")
    return val


def generate_replicate(job: ImageJob, output_path: Path) -> None:
    """Generate using Replicate Flux model (fastest, cheap)."""
    try:
        import replicate
    except ImportError:
        raise BackendError("Install: pip install replicate")

    token = _check_env("REPLICATE_API_TOKEN")
    model = "black-forest-labs/flux-schnell"

    prompt = job.get_prompt_text()
    print(f"  Calling Replicate {model}...", end=" ", flush=True)

    output = replicate.run(
        model,
        input={
            "prompt": prompt,
            "width": job.width,
            "height": job.height,
            "num_outputs": 1,
            "guidance_scale": 3.5,
            "num_inference_steps": 4 if "schnell" in model else 25,
        }
    )

    if output and len(output) > 0:
        image_url = str(output[0])
        print(f"downloading from {image_url[:50]}...")
        _download_url(image_url, output_path)
        print(f"  ✅ {output_path.name}")
    else:
        raise BackendError("Replicate returned no output")


def generate_openai(job: ImageJob, output_path: Path) -> None:
    """Generate using OpenAI DALL-E 3."""
    try:
        from openai import OpenAI
    except ImportError:
        raise BackendError("Install: pip install openai")

    key = _check_env("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    prompt = job.get_prompt_text()
    # DALL-E has size restrictions
    size = "1792x1024" if job.width > job.height else "1024x1792"
    if abs(job.width - job.height) < 200:
        size = "1024x1024"

    print(f"  Calling DALL-E 3 ({size})...", end=" ", flush=True)
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt[:4000],
        size=size,
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    if image_url:
        _download_url(image_url, output_path)
        print(f"  ✅ {output_path.name}")
    else:
        raise BackendError("OpenAI returned no URL")


def generate_together(job: ImageJob, output_path: Path) -> None:
    """Generate using Together AI."""
    try:
        import together
    except ImportError:
        raise BackendError("Install: pip install together")

    key = _check_env("TOGETHER_API_KEY")
    together.api_key = key

    prompt = job.get_prompt_text()
    print(f"  Calling Together FLUX.1-dev...", end=" ", flush=True)

    response = together.Image.create(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-dev",
        width=job.width,
        height=job.height,
        steps=25,
        n=1,
    )
    image_url = response.get("data", [{}])[0].get("url", "")
    if image_url:
        _download_url(image_url, output_path)
        print(f"  ✅ {output_path.name}")
    else:
        raise BackendError("Together returned no URL")


def _download_url(url: str, output_path: Path) -> None:
    """Download a URL to a file."""
    import urllib.request
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)


# ── Helpers ─────────────────────────────────────────────────────────────────

BACKENDS = {
    "replicate": generate_replicate,
    "openai": generate_openai,
    "together": generate_together,
}

BACKEND_DESCRIPTIONS = {
    "replicate": "Flux (fastest, ~$0.003/image)",
    "openai": "DALL-E 3 (best text, ~$0.04/image)",
    "together": "FLUX.1-dev (~$0.003/image)",
}


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="paper-craft-skills — Generate all example images"
    )
    parser.add_argument("--backend", choices=list(BACKENDS.keys()) + ["check"],
                        default="replicate",
                        help=f"Image generation backend (default: replicate)")
    parser.add_argument("--all", "-a", action="store_true",
                        help="Generate ALL example images")
    parser.add_argument("--list", action="store_true",
                        help="List available images and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be generated without calling API")
    parser.add_argument("images", nargs="*",
                        help="Image keys to generate (default: all if --all)")
    args = parser.parse_args()

    # Determine which jobs to run
    jobs_to_run: list[ImageJob] = []
    if args.list:
        print(f"\n{'='*55}")
        print("Available images to generate:")
        print(f"{'='*55}\n")
        for j in JOBS:
            print(f"  {j.key:<25} {j.title:<40} {j.width}x{j.height}")
        print(f"\nBackends:")
        for b, d in BACKEND_DESCRIPTIONS.items():
            print(f"  {b:<15} {d}")
        print(f"\n  --backend check    Check which images already exist")
        return

    if args.all:
        jobs_to_run = list(JOBS)
    elif args.images:
        for key in args.images:
            if key in JOB_MAP:
                jobs_to_run.append(JOB_MAP[key])
            else:
                print(f"  ⚠  Unknown image key: {key}. Use --list to see all.")
    else:
        print("Specify --all or a list of image keys. Use --list to see available images.")
        sys.exit(1)

    # Check mode
    if args.backend == "check":
        print(f"\n{'='*55}")
        print("Image Generation Check")
        print(f"{'='*55}\n")
        for j in jobs_to_run:
            output = ROOT / j.output_name
            prompt_ok = j.prompt_path.exists()
            image_ok = output.exists()
            print(f"  {j.key:<25} prompt={'✓' if prompt_ok else '✗'} image={'✓' if image_ok else '✗'} {output.name if image_ok else ''}")
        return

    # Dry run
    if args.dry_run:
        print(f"\n{'='*55}")
        print(f"DRY RUN — Would generate {len(jobs_to_run)} images:")
        print(f"{'='*55}\n")
        for j in jobs_to_run:
            output = ROOT / j.output_name
            prompt = j.get_prompt_text()
            print(f"  {j.key:<25} → {output}")
            print(f"  size: {j.width}x{j.height}  prompt: {len(prompt)} chars")
            print()
        print(f"  Backend: {args.backend}")
        print(f"  Total: {len(jobs_to_run)} images")
        return

    # Real generation
    generator = BACKENDS.get(args.backend)
    if not generator:
        print(f"Unknown backend: {args.backend}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"🎨 Generating {len(jobs_to_run)} images with {args.backend}")
    print(f"{'='*55}\n")

    success = 0
    failed = 0

    for j in jobs_to_run:
        output = ROOT / j.output_name
        if output.exists():
            print(f"  ⏭  {j.key}: already exists at {output}")
            success += 1
            continue

        print(f"\n  [{j.key}] {j.title}")
        print(f"  Output: {output}")

        if not j.prompt_path.exists():
            print(f"  ⚠  Prompt file not found: {j.prompt_path}")
            failed += 1
            continue

        try:
            generator(j, output)
            success += 1
        except BackendError as e:
            print(f"  ❌ {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ Unexpected error: {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"Results: {success} generated, {failed} failed")
    print(f"{'='*55}")

    if failed > 0:
        print(f"\n💡 Tip: Check API_KEY_GUIDE.md for setup instructions.")
        sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
