#!/usr/bin/env python3
"""
生成 HTML 文件，支持 base64 嵌入图片
"""

import os
import sys
import base64
import re
from pathlib import Path
from typing import Optional

try:
    import markdown
except ImportError:
    markdown = None


HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
               line-height: 1.8; max-width: 800px; margin: 0 auto; padding: 20px; color: #333; }}
        h1 {{ color: #1a1a1a; border-bottom: 2px solid #333; padding-bottom: 10px; }}
        h2 {{ color: #2c2c2c; margin-top: 30px; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 20px auto;
               border: 1px solid #ddd; border-radius: 4px; }}
        pre {{ background: #f5f5f5; padding: 15px; overflow-x: auto; border-radius: 4px; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
        blockquote {{ border-left: 4px solid #ddd; margin: 0; padding-left: 20px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
{content}
</body>
</html>'''


def get_mime_type(ext: str) -> str:
    """获取图片 MIME 类型"""
    types = {'.png': 'image/png', '.jpg': 'image/jpeg',
             '.jpeg': 'image/jpeg', '.gif': 'image/gif'}
    return types.get(ext.lower(), 'image/png')


def embed_image(img_path: Path) -> str:
    """将图片转换为 base64"""
    with open(img_path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    mime = get_mime_type(img_path.suffix)
    return f"data:{mime};base64,{data}"


def process_images(content: str, base_dir: Path) -> str:
    """处理 markdown 中的图片，转换为 base64"""
    def replace_img(match):
        alt, src = match.groups()
        img_path = base_dir / src
        if img_path.exists():
            b64 = embed_image(img_path)
            return f'![{alt}]({b64})'
        return match.group(0)

    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_img, content)


def md_to_html(md_content: str) -> str:
    """Markdown 转 HTML"""
    if markdown:
        return markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
    # 简单的备选转换
    html = md_content
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.M)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.M)
    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.M)
    html = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1">', html)
    html = re.sub(r'\n\n', '</p><p>', html)
    return f'<p>{html}</p>'


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_html.py <md_path> [output.html]")
        sys.exit(1)

    md_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "article.html")

    if not md_path.exists():
        print(f"Error: File not found: {md_path}")
        sys.exit(1)

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 处理图片为 base64
    base_dir = md_path.parent
    content = process_images(content, base_dir)

    # 转换为 HTML
    html_content = md_to_html(content)

    # 提取标题
    title_match = re.search(r'^#\s+(.+)$', content, re.M)
    title = title_match.group(1) if title_match else "Article"

    # 生成完整 HTML
    html = HTML_TEMPLATE.format(title=title, content=html_content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Generated: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
