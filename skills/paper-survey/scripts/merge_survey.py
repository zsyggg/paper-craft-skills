#!/usr/bin/env python3
"""
Assemble paper-survey outputs: comparison matrix, timeline, and survey artifacts.

Usage:
    python3 merge_survey.py /path/to/survey-slug
    python3 merge_survey.py /path/to/survey-slug --format comparison-figures
    python3 merge_survey.py /path/to/survey-slug --format survey-deck
    python3 merge_survey.py /path/to/survey-slug --format survey-article
    python3 merge_survey.py /path/to/survey-slug --format all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class PaperSummary:
    slug: str
    title: str
    authors: str
    year: int
    core_idea: str
    key_design: str
    key_result: str
    limitation: str
    code_url: str = ""


@dataclass
class ComparisonDimension:
    name: str
    label: str
    papers: dict[str, str]  # paper_slug -> description


@dataclass
class ComparisonMatrix:
    title: str
    papers: list[PaperSummary]
    dimensions: list[ComparisonDimension]
    timeline: list[dict]  # {year, paper, milestone}


@dataclass
class SurveyReport:
    matrix: ComparisonMatrix
    paper_count: int
    dimension_count: int
    has_timeline: bool
    prompts_ready: int
    images_ready: int
    errors: list[str] = field(default_factory=list)


# ── File Operations ─────────────────────────────────────────────────────────

def load_paper_summaries(papers_dir: Path) -> list[PaperSummary]:
    """Load per-paper markdown analysis files."""
    papers: list[PaperSummary] = []
    if not papers_dir.exists():
        return papers

    for md_file in sorted(papers_dir.glob("*.md")):
        with open(md_file, encoding="utf-8") as f:
            content = f.read()

        def extract_section(pattern: str, default: str = "") -> str:
            m = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            return m.group(1).strip() if m else default

        # Extract from YAML-like frontmatter or section headers
        title = extract_section(r"(?i)^#+\s+(.+)$", md_file.stem)
        core = extract_section(r"(?i)(?:核心|core)\s*(?:创新|idea|contribution)[：:]\s*(.+?)(?=\n\n|\n#{1,3}\s)", "")
        design = extract_section(r"(?i)(?:关键|key)\s*(?:设计|design)[：:]\s*(.+?)(?=\n\n|\n#{1,3}\s)", "")
        result = extract_section(r"(?i)(?:核心|key)\s*(?:结果|result)[：:]\s*(.+?)(?=\n\n|\n#{1,3}\s)", "")
        limit = extract_section(r"(?i)(?:局限|limitation)[：:]\s*(.+?)(?=\n\n|\n#{1,3}\s)", "")
        code = extract_section(r"(?i)(?:代码|code)[：:]\s*(.+?)(?=\n)", "")

        papers.append(PaperSummary(
            slug=md_file.stem,
            title=title,
            authors="",
            year=0,
            core_idea=core,
            key_design=design,
            key_result=result,
            limitation=limit,
            code_url=code,
        ))
    return papers


def load_comparison_matrix(analysis_dir: Path) -> Optional[ComparisonMatrix]:
    """Load a pre-built comparison-matrix.json if it exists."""
    matrix_path = analysis_dir / "comparison-matrix.json"
    if not matrix_path.exists():
        return None
    with open(matrix_path, encoding="utf-8") as f:
        data = json.load(f)
    return ComparisonMatrix(**data)


def count_prompts(survey_dir: Path) -> int:
    prompts_dir = survey_dir / "prompts"
    if not prompts_dir.exists():
        return 0
    return len(list(prompts_dir.glob("*.md")))


def count_images(survey_dir: Path) -> int:
    images_dir = survey_dir / "images"
    if not images_dir.exists():
        return 0
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return len([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


# ── Report Generation ───────────────────────────────────────────────────────

def generate_report(survey_dir: Path) -> SurveyReport:
    """Analyze a survey directory and produce a structured report."""
    papers = load_paper_summaries(survey_dir / "papers")
    matrix = load_comparison_matrix(survey_dir / "analysis")

    if not matrix:
        # Build a minimal matrix from available papers
        dimensions = []
    else:
        dimensions = matrix.dimensions

    prompts_ready = count_prompts(survey_dir)
    images_ready = count_images(survey_dir)
    timeline_path = survey_dir / "analysis" / "timeline.json"
    has_timeline = timeline_path.exists()

    return SurveyReport(
        matrix=matrix or ComparisonMatrix(
            title=survey_dir.name,
            papers=papers,
            dimensions=dimensions,
            timeline=[],
        ),
        paper_count=len(papers),
        dimension_count=len(dimensions),
        has_timeline=has_timeline,
        prompts_ready=prompts_ready,
        images_ready=images_ready,
    )


# ── Output Formatters ───────────────────────────────────────────────────────

def format_text_report(report: SurveyReport) -> str:
    lines = [
        f"{'='*55}",
        f"  Survey Report — {report.matrix.title}",
        f"{'='*55}",
        f"",
        f"  Papers: {report.paper_count}",
        f"  Comparison Dimensions: {report.dimension_count}",
        f"  Timeline: {'Yes' if report.has_timeline else 'No'}",
        f"  Prompts written: {report.prompts_ready}",
        f"  Images generated: {report.images_ready}",
        f"",
    ]

    if report.matrix.papers:
        lines.append(f"  ── Papers ──")
        for p in report.matrix.papers:
            lines.append(f"    {p.slug}: {p.title[:60]}")
            if p.core_idea:
                lines.append(f"      Core: {p.core_idea[:80]}")

    if report.matrix.dimensions:
        lines.append(f"")
        lines.append(f"  ── Comparison Dimensions ──")
        for d in report.matrix.dimensions:
            lines.append(f"    {d.label} ({d.name})")
            for slug, desc in d.papers.items():
                lines.append(f"      {slug}: {desc[:60]}")

    if report.prompts_ready == 0 and report.images_ready == 0:
        lines.append(f"")
        lines.append(f"  ⚠  No prompts or images yet. Run the survey generation workflow.")

    lines.append(f"")
    lines.append(f"{'='*55}")
    return "\n".join(lines)


def format_json_report(report: SurveyReport) -> str:
    return json.dumps(asdict(report), indent=2, ensure_ascii=False)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble paper-survey outputs."
    )
    parser.add_argument("survey_dir", help="Survey directory")
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Report format (default: text)"
    )
    args = parser.parse_args()

    survey_dir = Path(args.survey_dir).expanduser().resolve()
    if not survey_dir.exists():
        raise SystemExit(f"[ERROR] Survey directory not found: {survey_dir}")

    report = generate_report(survey_dir)

    if args.format == "json":
        print(format_json_report(report))
    else:
        print(format_text_report(report))

    # Non-zero exit if nothing generated yet
    if report.prompts_ready == 0 and report.images_ready == 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
