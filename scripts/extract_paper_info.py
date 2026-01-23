#!/usr/bin/env python3
"""
提取论文元数据信息
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List


def extract_title(text: str) -> str:
    """提取论文标题"""
    lines = text.split('\n')
    for line in lines[:20]:
        line = line.strip()
        # 跳过空行和太短的行
        if len(line) < 10:
            continue
        # 跳过明显不是标题的行
        if line.startswith('#'):
            return line.lstrip('#').strip()
        if len(line) < 150 and not line.startswith('http'):
            return line
    return "Unknown Title"


def extract_abstract(text: str) -> str:
    """提取摘要"""
    # 查找 Abstract 部分
    patterns = [
        r'(?i)abstract\s*\n(.*?)(?=\n\s*(?:introduction|1\.|keywords))',
        r'(?i)abstract[:\s]+(.*?)(?=\n\n)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            if len(abstract) > 50:
                return abstract[:1000]
    return ""


def extract_sections(text: str) -> List[str]:
    """提取章节标题"""
    sections = []
    patterns = [
        r'^#+\s+(.+)$',  # Markdown 标题
        r'^(\d+\.?\s+[A-Z][^.]+)$',  # 数字编号标题
        r'^([A-Z][A-Z\s]+)$',  # 全大写标题
    ]
    for line in text.split('\n'):
        line = line.strip()
        for pattern in patterns:
            if re.match(pattern, line) and 5 < len(line) < 100:
                sections.append(line.lstrip('#').strip())
                break
    return sections[:30]


def extract_figures(text: str, images_dir: Path) -> List[Dict]:
    """提取图片信息"""
    figures = []

    # 从 markdown 中提取图片引用
    img_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    for match in re.finditer(img_pattern, text):
        caption, path = match.groups()
        figures.append({"caption": caption, "path": path})

    # 从 images 目录获取实际图片
    if images_dir.exists():
        for img_file in sorted(images_dir.glob("*")):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
                if not any(f["path"].endswith(img_file.name) for f in figures):
                    figures.append({
                        "caption": img_file.stem,
                        "path": f"images/{img_file.name}"
                    })

    return figures


def extract_paper_info(md_path: Path, images_dir: Path) -> Dict:
    """提取论文完整信息"""
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return {
        "title": extract_title(text),
        "abstract": extract_abstract(text),
        "sections": extract_sections(text),
        "figures": extract_figures(text, images_dir),
        "word_count": len(text.split()),
        "char_count": len(text)
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_paper_info.py <md_path> [output.json]")
        sys.exit(1)

    md_path = Path(sys.argv[1])
    output_path = sys.argv[2] if len(sys.argv) > 2 else "paper_info.json"

    if not md_path.exists():
        print(f"Error: File not found: {md_path}")
        sys.exit(1)

    images_dir = md_path.parent / "images"
    info = extract_paper_info(md_path, images_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"Extracted: {info['title'][:50]}...")
    print(f"Sections: {len(info['sections'])}")
    print(f"Figures: {len(info['figures'])}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
