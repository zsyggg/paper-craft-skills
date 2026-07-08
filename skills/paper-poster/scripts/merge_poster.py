#!/usr/bin/env python3
"""
Merge high-resolution poster image and produce print-ready PDF.
Supports upscaling, tiling for large posters, and quality checks.

Usage:
    python3 merge_poster.py /path/to/poster-dir
    python3 merge_poster.py /path/to/poster-dir --check
    python3 merge_poster.py /path/to/poster-dir --dpi 600
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


# ── Constants ──────────────────────────────────────────────────────────────

POSTER_SIZES = {
    "conference-wide": (48, 36),
    "defense-poster": (36, 48),
    "research-showcase": (48, 36),
}
TARGET_DPI = 300


# ── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class PosterConfig:
    preset: str
    width_in: float
    height_in: float
    dpi: int = TARGET_DPI

    @property
    def width_px(self) -> int:
        return int(self.width_in * self.dpi)

    @property
    def height_px(self) -> int:
        return int(self.height_in * self.dpi)


@dataclass
class QualityReport:
    passed: bool = True
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ── Checks ──────────────────────────────────────────────────────────────────

def check_poster_image(img_path: Path, config: PosterConfig) -> QualityReport:
    report = QualityReport()
    if not img_path.exists():
        report.errors.append("No poster image found")
        report.passed = False
        return report

    with Image.open(img_path) as img:
        w, h = img.size
        report.checks.append(f"Image: {w}x{h} px")

        if w < config.width_px * 0.5:
            report.warnings.append(
                f"Low resolution ({w}px vs target {config.width_px}px)"
            )

        actual_ratio = w / h
        expected_ratio = config.width_in / config.height_in
        if abs(actual_ratio - expected_ratio) > 0.05:
            report.warnings.append(
                f"Aspect ratio mismatch: {actual_ratio:.2f} vs {expected_ratio:.2f}"
            )

    return report


# ── Upscaling ───────────────────────────────────────────────────────────────

def upscale_image(img_path: Path, config: PosterConfig, output: Path) -> None:
    with Image.open(img_path) as img:
        w, h = img.size
        target_w = config.width_px
        target_h = config.height_px

        if w >= target_w and h >= target_h:
            print(f"  Already meets {config.dpi}dpi target ({target_w}x{target_h})")
            img.save(output)
            return

        print(f"  Upscaling {w}x{h} to {target_w}x{target_h}")
        resized = img.resize((target_w, target_h), Image.LANCZOS)
        resized.save(output, dpi=(config.dpi, config.dpi))
        print(f"  Saved: {output}")


def make_poster_pdf(image_path: Path, config: PosterConfig, output: Path) -> None:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        rgb.save(output, dpi=(config.dpi, config.dpi))
    print(f"  PDF: {output} ({config.width_px}x{config.height_px} @ {config.dpi}dpi)")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble paper-poster output.")
    parser.add_argument("poster_dir", help="Poster directory")
    parser.add_argument("--check", action="store_true", help="Quality check only")
    parser.add_argument("--dpi", type=int, default=TARGET_DPI, help=f"Target DPI (default: {TARGET_DPI})")
    args = parser.parse_args()

    poster_dir = Path(args.poster_dir).expanduser().resolve()
    if not poster_dir.exists():
        raise SystemExit(f"[ERROR] Poster directory not found: {poster_dir}")

    brief_path = poster_dir / "poster-brief.md"
    if not brief_path.exists():
        raise SystemExit(f"[ERROR] poster-brief.md not found in {poster_dir}")

    with open(brief_path, encoding="utf-8") as f:
        brief = f.read()

    preset = "conference-wide"
    for p in POSTER_SIZES:
        if p in brief:
            preset = p
            break

    w, h = POSTER_SIZES[preset]
    config = PosterConfig(preset=preset, width_in=w, height_in=h, dpi=args.dpi)
    print(f"Poster: {preset} ({w}x{h} in @ {args.dpi}dpi = {config.width_px}x{config.height_px} px)")

    img_exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted([p for p in poster_dir.glob("*") if p.suffix.lower() in img_exts])
    if not images:
        img_dir = poster_dir / "images"
        if img_dir.exists():
            images = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in img_exts])
    if not images:
        raise SystemExit(f"[ERROR] No poster image found in {poster_dir}")

    source = images[0]

    report = check_poster_image(source, config)
    for c in report.checks:
        print(f"  [OK] {c}")
    for w in report.warnings:
        print(f"  [WARN] {w}")
    if args.check:
        return

    print(f"\nProcessing {source.name}...")
    output_dir = poster_dir / "output"
    output_dir.mkdir(exist_ok=True)

    upscaled = output_dir / "poster-print.png"
    upscale_image(source, config, upscaled)

    pdf_path = output_dir / "poster.pdf"
    make_poster_pdf(upscaled, config, pdf_path)

    print(f"\n{'='*50}")
    print(f"Poster ready:")
    print(f"  Print:  {upscaled}")
    print(f"  PDF:    {pdf_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
