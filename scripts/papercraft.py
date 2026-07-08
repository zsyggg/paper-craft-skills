#!/usr/bin/env python3
"""
papercraft — Unified CLI for paper-craft-skills.

Dispatch to all 5 skills from a single command:
  papercraft comic <paper>          → paper-comic
  papercraft analyze <paper>        → paper-analyzer
  papercraft deck <paper>           → paper-deck
  papercraft survey <paper1> <p2>…   → paper-survey
  papercraft poster <paper>         → paper-poster
  papercraft summary <paper>        → paper-summary

Usage:
  papercraft comic https://arxiv.org/abs/1706.03762 --style sketchnote
  papercraft deck paper.pdf --style journal-minimal --slides 12
  papercraft survey p1.pdf p2.pdf --format survey-article
  papercraft list                    → list available skills
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SKILL_DESCRIPTIONS = {
    "comic":  "🎨  Paper → AIGC method figures (sketchnote / paper-figure)",
    "analyze": "📄  Paper → deep analysis HTML article (storytelling / academic / concise)",
    "deck":   "🖼️  Paper → visual slide deck PPTX/PDF (4 style presets)",
    "survey": "🔬  Multi-paper → comparison figures / survey deck / survey article",
    "poster": "📋  Paper → AIGC conference poster (conference-wide / defense / showcase)",
    "summary":"📊  Paper → one-page visual summary / infographic / card",
}


def cmd_list(args: argparse.Namespace) -> None:
    print(f"\n{'='*55}")
    print("  paper-craft-skills — available commands")
    print(f"{'='*55}\n")
    for name, desc in SKILL_DESCRIPTIONS.items():
        print(f"  papercraft {name:<10}  {desc}")
    print(f"\n  papercraft list          Show this help")
    print(f"  papercraft help <skill>  Show skill details")
    print()


def cmd_help(args: argparse.Namespace) -> None:
    skill = args.skill_name
    if skill in SKILL_DESCRIPTIONS:
        print(f"\n  papercraft {skill}")
        print(f"  {SKILL_DESCRIPTIONS[skill]}")
        print(f"\n  Usage examples:")
        if skill == "comic":
            print(f"    papercraft comic https://arxiv.org/abs/1706.03762")
            print(f"    papercraft comic paper.pdf --style sketchnote --language Chinese")
        elif skill == "analyze":
            print(f"    papercraft analyze https://arxiv.org/abs/1706.03762")
            print(f"    papercraft analyze paper.pdf --style academic")
        elif skill == "deck":
            print(f"    papercraft deck https://arxiv.org/abs/1706.03762")
            print(f"    papercraft deck paper.pdf --style liquid-glass --slides 12")
        elif skill == "survey":
            print(f"    papercraft survey paper1.pdf paper2.pdf paper3.pdf")
            print(f"    papercraft survey --topic 'topic' --papers p1.pdf p2.pdf")
        elif skill == "poster":
            print(f"    papercraft poster https://arxiv.org/abs/1706.03762")
            print(f"    papercraft poster paper.pdf --style defense-poster")
        elif skill == "summary":
            print(f"    papercraft summary https://arxiv.org/abs/1706.03762")
            print(f"    papercraft summary paper.pdf --mode infographic --language Chinese")
        print()
    else:
        print(f"Unknown skill: {skill}. Use 'papercraft list' to see all skills.")


def cmd_comic(args: argparse.Namespace) -> None:
    papers = " ".join(args.papers)
    style = f" --style {args.style}" if args.style else ""
    lang = f" --language {args.language}" if args.language else ""
    print(f"\n  🎨 paper-comic")
    print(f"  {'='*40}")
    print(f"  Paper: {papers}")
    print(f"  Style: {args.style or 'sketchnote (default)'}")
    print(f"  Language: {args.language or 'auto'}")
    print(f"\n  ▶  Tell your AI agent: /paper-comic {papers}{style}{lang}")
    print()


def cmd_analyze(args: argparse.Namespace) -> None:
    papers = " ".join(args.papers)
    style = f" --style {args.style}" if args.style else ""
    print(f"\n  📄 paper-analyzer")
    print(f"  {'='*40}")
    print(f"  Paper: {papers}")
    print(f"  Style: {args.style or 'academic (default)'}")
    print(f"\n  ▶  Tell your AI agent: /paper-analyzer {papers}{style}")
    print()


def cmd_deck(args: argparse.Namespace) -> None:
    papers = " ".join(args.papers)
    style = f" --style {args.style}" if args.style else ""
    slides = f" --slides {args.slides}" if args.slides else ""
    print(f"\n  🖼️  paper-deck")
    print(f"  {'='*40}")
    print(f"  Paper: {papers}")
    print(f"  Style: {args.style or 'journal-minimal (default)'}")
    if args.slides:
        print(f"  Slides: {args.slides}")
    print(f"\n  ▶  Tell your AI agent: /paper-deck {papers}{style}{slides}")
    print()


def cmd_survey(args: argparse.Namespace) -> None:
    papers_list = args.papers if args.papers else []
    topic = args.topic or ""
    fmt = f" --format {args.format}" if args.format else ""

    print(f"\n  🔬 paper-survey")
    print(f"  {'='*40}")
    if topic:
        print(f"  Topic: {topic}")
    if papers_list:
        print(f"  Papers: {' '.join(papers_list)}")
    print(f"  Format: {args.format or 'auto'}")
    print(f"\n  ▶  Tell your AI agent: /paper-survey {' '.join(papers_list)}{fmt}")
    print()


def cmd_poster(args: argparse.Namespace) -> None:
    papers = " ".join(args.papers)
    style = f" --style {args.style}" if args.style else ""
    print(f"\n  📋 paper-poster")
    print(f"  {'='*40}")
    print(f"  Paper: {papers}")
    print(f"  Style: {args.style or 'conference-wide (default)'}")
    print(f"\n  ▶  Tell your AI agent: /paper-poster {papers}{style}")
    print()


def cmd_summary(args: argparse.Namespace) -> None:
    papers = " ".join(args.papers)
    mode = f" --mode {args.mode}" if args.mode else ""
    lang = f" --language {args.language}" if args.language else ""
    print(f"\n  📊 paper-summary")
    print(f"  {'='*40}")
    print(f"  Paper: {papers}")
    print(f"  Mode: {args.mode or 'infographic (default)'}")
    print(f"  Language: {args.language or 'auto'}")
    print(f"\n  ▶  Tell your AI agent: /paper-summary {papers}{mode}{lang}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="paper-craft-skills — unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  papercraft comic https://arxiv.org/abs/1706.03762 --style sketchnote
  papercraft deck paper.pdf --style liquid-glass --slides 12
  papercraft survey p1.pdf p2.pdf p3.pdf --format survey-article
  papercraft list
  papercraft help deck
        """
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # list
    sub.add_parser("list", help="List all available skills")

    # help <skill>
    help_p = sub.add_parser("help", help="Show help for a specific skill")
    help_p.add_argument("skill_name", help="Skill name (comic, analyze, deck, survey, poster, summary)")

    # comic
    comic_p = sub.add_parser("comic", help="Paper → AIGC method figures")
    comic_p.add_argument("papers", nargs="+", help="Paper URL or PDF path")
    comic_p.add_argument("--style", choices=["sketchnote", "paper-figure"], help="Visual style")
    comic_p.add_argument("--language", choices=["Chinese", "English"], help="Language")

    # analyze
    analyze_p = sub.add_parser("analyze", help="Paper → deep analysis article")
    analyze_p.add_argument("papers", nargs="+", help="Paper URL or PDF path")
    analyze_p.add_argument("--style", choices=["storytelling", "academic", "concise"], help="Writing style")

    # deck
    deck_p = sub.add_parser("deck", help="Paper → visual slide deck")
    deck_p.add_argument("papers", nargs="+", help="Paper URL or PDF path")
    deck_p.add_argument("--style", choices=["journal-minimal", "business-research", "warm-notes", "liquid-glass"], help="Slide style")
    deck_p.add_argument("--slides", type=int, help="Number of slides")

    # survey
    survey_p = sub.add_parser("survey", help="Multi-paper → comparison survey")
    survey_p.add_argument("papers", nargs="*", help="Paper URLs or PDF paths")
    survey_p.add_argument("--topic", help="Research topic (alternative to papers)")
    survey_p.add_argument("--format", choices=["comparison-figures", "survey-deck", "survey-article"], help="Output format")

    # poster
    poster_p = sub.add_parser("poster", help="Paper → AIGC conference poster")
    poster_p.add_argument("papers", nargs="+", help="Paper URL or PDF path")
    poster_p.add_argument("--style", choices=["conference-wide", "defense-poster", "research-showcase"], help="Poster style")

    # summary
    summary_p = sub.add_parser("summary", help="Paper → one-page visual summary")
    summary_p.add_argument("papers", nargs="+", help="Paper URL or PDF path")
    summary_p.add_argument("--mode", choices=["infographic", "card-summary", "timeline-figure"], help="Summary mode")
    summary_p.add_argument("--language", choices=["Chinese", "English"], help="Language")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    dispatch = {
        "list": cmd_list,
        "help": cmd_help,
        "comic": cmd_comic,
        "analyze": cmd_analyze,
        "deck": cmd_deck,
        "survey": cmd_survey,
        "poster": cmd_poster,
        "summary": cmd_summary,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
