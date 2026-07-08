#!/usr/bin/env python3
"""
paper-craft-skills — Batch Image Generation Script

Generates all example images by printing the prompts ready for copy-paste
into any image generation backend (Flux, DALL-E, Midjourney, Codex imagegen).

Usage:
    python3 scripts/batch_generate.py                       # Show all prompts
    python3 scripts/batch_generate.py --list                 # List all available
    python3 scripts/batch_generate.py poster                # Just poster prompts
    python3 scripts/batch_generate.py survey                # Just survey prompts
    python3 scripts/batch_generate.py summary               # Just summary prompts
    python3 scripts/batch_generate.py --html                # Generate HTML dashboard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           paper-craft-skills — Batch Generator              ║
║                                                             ║
║  Copy any prompt below into your preferred image gen tool   ║
║  Supported: Flux · DALL-E 3 · Midjourney · Codex imagegen  ║
╚══════════════════════════════════════════════════════════════╝
"""

PROMPTS = {
    "summary-infographic": {
        "title": "Paper Summary — Infographic",
        "file": "examples/paper-summary/attention-is-all-you-need/prompt-infographic.md",
        "description": "Attention Is All You Need — vertical 2:3 infographic",
        "tags": ["summary", "infographic", "attention"],
    },
    "poster-conference": {
        "title": "Conference Poster — Transformer",
        "file": "examples/paper-poster/attention-is-all-you-need/prompt-poster.md",
        "description": "Attention Is All You Need — 48x36 conference poster",
        "tags": ["poster", "conference-wide", "attention"],
    },
    "survey-overview": {
        "title": "Survey — Overview Comparison",
        "file": "examples/paper-survey/transformer-position-encoding/prompt-overview.md",
        "description": "4 position encoding methods — overview comparison",
        "tags": ["survey", "comparison", "position-encoding"],
    },
    "survey-architecture": {
        "title": "Survey — Architecture Comparison",
        "file": "examples/paper-survey/transformer-position-encoding/prompt-architecture.md",
        "description": "Position injection method comparison",
        "tags": ["survey", "architecture", "position-encoding"],
    },
    "survey-experiments": {
        "title": "Survey — Experiments Comparison",
        "file": "examples/paper-survey/transformer-position-encoding/prompt-experiments.md",
        "description": "Extrapolation benchmark comparison",
        "tags": ["survey", "experiments", "position-encoding"],
    },
    "survey-timeline": {
        "title": "Survey — Evolution Timeline",
        "file": "examples/paper-survey/transformer-position-encoding/prompt-timeline.md",
        "description": "Research evolution 2017→2021",
        "tags": ["survey", "timeline", "position-encoding"],
    },
}


def find_project_root() -> Path:
    """Find the project root by looking for README.md."""
    candidates = [
        Path(__file__).resolve().parent.parent,
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "README.md").exists():
            return c
    return Path.cwd()


def print_prompt(key: str, prompt: dict, root: Path) -> None:
    """Print a single prompt with header."""
    file_path = root / prompt["file"]
    if not file_path.exists():
        print(f"\n  ⚠  File not found: {file_path}")
        return

    print(f"\n{'='*60}")
    print(f"  📋 {prompt['title']}")
    print(f"  {prompt['description']}")
    print(f"  📁 {prompt['file']}")
    print(f"{'='*60}\n")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Extract the prompt content (skip markdown headers)
    in_prompt = False
    for line in content.split("\n"):
        if line.strip().startswith("```markdown") or line.strip().startswith("```text"):
            in_prompt = True
            continue
        if line.strip().startswith("```") and in_prompt:
            in_prompt = False
            continue
        if in_prompt:
            print(line)

    print(f"\n{'─'*60}")


def generate_html_dashboard(root: Path, output: Optional[Path] = None) -> Path:
    """Generate an HTML dashboard with all prompts for easy copying."""
    if output is None:
        output = root / "generation-dashboard.html"

    cards = []
    for key, prompt in PROMPTS.items():
        file_path = root / prompt["file"]
        prompt_content = ""
        if file_path.exists():
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            # Extract prompt from code block
            in_block = False
            for line in content.split("\n"):
                if line.strip().startswith("```markdown") or line.strip().startswith("```text"):
                    in_block = True
                    continue
                if line.strip().startswith("```") and in_block:
                    in_block = False
                    continue
                if in_block:
                    prompt_content += line + "\n"

        tags_html = " ".join(f'<span class="tag">{t}</span>' for t in prompt["tags"])
        prompt_escaped = prompt_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        cards.append(f"""
        <div class="card">
            <div class="card-header">
                <h2>{prompt['title']}</h2>
                <div class="tags">{tags_html}</div>
            </div>
            <p class="desc">{prompt['description']}</p>
            <textarea class="prompt-box" readonly onclick="this.select()">{prompt_escaped}</textarea>
            <button class="copy-btn" onclick="copyPrompt(this)">📋 Copy</button>
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>paper-craft-skills — Generation Dashboard</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a; color: #e2e8f0; padding: 2rem; max-width: 1200px; margin: 0 auto;
}}
h1 {{ font-size: 2rem; margin-bottom: 0.5rem; color: #60a5fa; }}
.subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 1.5rem; }}
.card {{
    background: #1e293b; border-radius: 12px; padding: 1.5rem;
    border: 1px solid #334155; transition: border-color 0.2s;
}}
.card:hover {{ border-color: #60a5fa; }}
.card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }}
.card-header h2 {{ font-size: 1.1rem; color: #f1f5f9; margin: 0; }}
.tags {{ display: flex; gap: 0.4rem; }}
.tag {{ background: #334155; color: #94a3b8; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; }}
.desc {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem; }}
.prompt-box {{
    width: 100%; min-height: 120px; background: #0f172a; color: #e2e8f0;
    border: 1px solid #334155; border-radius: 8px; padding: 1rem;
    font-family: "SF Mono", "Fira Code", monospace; font-size: 0.8rem; line-height: 1.5;
    resize: vertical; white-space: pre-wrap; word-wrap: break-word;
}}
.copy-btn {{
    margin-top: 0.5rem; background: #2563eb; color: white; border: none;
    padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer; font-size: 0.85rem;
    transition: background 0.2s;
}}
.copy-btn:hover {{ background: #3b82f6; }}
.copy-btn.copied {{ background: #16a34a; }}
.instructions {{
    background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem;
    border: 1px solid #334155;
}}
.instructions ol {{ margin-left: 1.5rem; line-height: 2; }}
.instructions code {{ background: #334155; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; }}
@media (max-width: 600px) {{ .grid {{ grid-template-columns: 1fr; }} body {{ padding: 1rem; }} }}
</style>
</head>
<body>
<h1>🎨 paper-craft-skills</h1>
<p class="subtitle">Image Generation Dashboard — Click any prompt box to select, then copy</p>

<div class="instructions">
    <h3>📖 How to use</h3>
    <ol>
        <li>Click a prompt box below to select all text (or use <code>Copy</code> button)</li>
        <li>Paste into your preferred image generation tool:
            <b>Flux</b> · <b>DALL-E 3</b> · <b>Midjourney</b> · <b>Codex imagegen</b>
        </li>
        <li>Save the output to the corresponding <code>images/</code> directory</li>
        <li>Use <code>merge_deck.py</code> / <code>merge_poster.py</code> for final assembly</li>
    </ol>
</div>

<div class="grid">
    {''.join(cards)}
</div>

<script>
function copyPrompt(btn) {{
    const textarea = btn.previousElementSibling;
    textarea.select();
    navigator.clipboard.writeText(textarea.value).then(() => {{
        btn.textContent = '✅ Copied!';
        btn.classList.add('copied');
        setTimeout(() => {{ btn.textContent = '📋 Copy'; btn.classList.remove('copied'); }}, 2000);
    }});
}}
</script>
</body>
</html>"""

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  ✅ Dashboard generated: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="paper-craft-skills batch image generation"
    )
    parser.add_argument("filter", nargs="?", default="all",
                        help="Filter: all / poster / survey / summary / <prompt-key>")
    parser.add_argument("--list", action="store_true", help="List all available prompts")
    parser.add_argument("--html", action="store_true", help="Generate HTML dashboard")
    args = parser.parse_args()

    root = find_project_root()

    if args.html:
        generate_html_dashboard(root)
        return

    print(BANNER)

    if args.list:
        print(f"\n  Available prompts ({len(PROMPTS)}):\n")
        for key, prompt in PROMPTS.items():
            tags = ", ".join(prompt["tags"])
            print(f"    {key:<25}  {prompt['title']:<35}  [{tags}]")
        print()
        return

    # Filter prompts
    selected = {}
    if args.filter == "all":
        selected = PROMPTS
    else:
        for key, prompt in PROMPTS.items():
            if args.filter in key or args.filter in prompt["tags"]:
                selected[key] = prompt

    if not selected:
        print(f"  No prompts matching '{args.filter}'. Use --list to see all.")
        sys.exit(1)

    print(f"  Showing {len(selected)} prompt(s). Copy into your image gen tool.\n")

    for key, prompt in selected.items():
        print_prompt(key, prompt, root)

    print(f"\n  💡 Tip: Use --html to generate an interactive dashboard with copy buttons.\n")


if __name__ == "__main__":
    main()
