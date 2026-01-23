#!/usr/bin/env python3
"""
PDF转换脚本 - 支持多种转换方式
1. 读取 MinerU 桌面应用输出（推荐）
2. 使用 PyMuPDF + 页面截图（备选）
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, List


def find_mineru_output(pdf_name: str, mineru_dir: str = "~/MinerU") -> Optional[Path]:
    """查找 MinerU 桌面应用的输出目录"""
    mineru_path = Path(mineru_dir).expanduser()
    if not mineru_path.exists():
        return None

    # MinerU 输出目录格式: filename.pdf-uuid
    for item in mineru_path.iterdir():
        if item.is_dir() and item.name.startswith(pdf_name):
            # 检查是否有 full.md 文件
            if (item / "full.md").exists():
                return item
    return None


def copy_mineru_output(mineru_dir: Path, output_dir: Path) -> Dict:
    """复制 MinerU 输出到目标目录"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 复制 markdown 文件
    md_src = mineru_dir / "full.md"
    md_dst = output_dir / "converted.md"
    shutil.copy2(md_src, md_dst)

    # 复制图片目录
    images_src = mineru_dir / "images"
    images_dst = output_dir / "images"
    if images_src.exists():
        if images_dst.exists():
            shutil.rmtree(images_dst)
        shutil.copytree(images_src, images_dst)

    # 统计信息
    image_count = len(list(images_dst.glob("*"))) if images_dst.exists() else 0
    md_size = md_dst.stat().st_size if md_dst.exists() else 0

    return {
        "method": "mineru",
        "markdown_path": str(md_dst),
        "images_dir": str(images_dst),
        "image_count": image_count,
        "markdown_size": md_size
    }


def convert_with_pymupdf(pdf_path: Path, output_dir: Path) -> Dict:
    """使用 PyMuPDF 转换（备选方案）"""
    try:
        import fitz
    except ImportError:
        print("Error: PyMuPDF not installed. Run: pip install PyMuPDF")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    doc = fitz.open(pdf_path)
    markdown_lines = [f"# {pdf_path.stem}\n"]
    image_count = 0

    for page_num, page in enumerate(doc, 1):
        # 提取文本
        text = page.get_text()
        if text.strip():
            markdown_lines.append(f"\n## Page {page_num}\n")
            markdown_lines.append(text)

        # 提取嵌入图片
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            if base_image and len(base_image["image"]) > 1024:
                ext = base_image["ext"]
                img_name = f"page{page_num}_img{img_idx}.{ext}"
                img_path = images_dir / img_name
                with open(img_path, "wb") as f:
                    f.write(base_image["image"])
                image_count += 1

        # 如果没有嵌入图片，截取整页作为图片
        if not page.get_images() and page_num <= 10:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_name = f"page{page_num}_full.png"
            pix.save(str(images_dir / img_name))
            markdown_lines.append(f"\n![Page {page_num}](images/{img_name})\n")
            image_count += 1

    doc.close()

    # 保存 markdown
    md_path = output_dir / "converted.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))

    return {
        "method": "pymupdf",
        "markdown_path": str(md_path),
        "images_dir": str(images_dir),
        "image_count": image_count,
        "markdown_size": md_path.stat().st_size
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_pdf.py <pdf_path> [output_dir]")
        print("\nOptions:")
        print("  pdf_path   - PDF file path")
        print("  output_dir - Output directory (default: ./output)")
        print("\nThe script will:")
        print("  1. First look for MinerU desktop app output")
        print("  2. Fall back to PyMuPDF if MinerU output not found")
        sys.exit(1)

    pdf_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "./output").resolve()

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"PDF: {pdf_path}")
    print(f"Output: {output_dir}")

    # 尝试查找 MinerU 输出
    pdf_name = pdf_path.stem
    mineru_output = find_mineru_output(pdf_name)

    if mineru_output:
        print(f"\nFound MinerU output: {mineru_output}")
        result = copy_mineru_output(mineru_output, output_dir)
    else:
        print("\nMinerU output not found, using PyMuPDF...")
        result = convert_with_pymupdf(pdf_path, output_dir)

    print(f"\nConversion complete!")
    print(f"  Method: {result['method']}")
    print(f"  Markdown: {result['markdown_path']}")
    print(f"  Images: {result['image_count']} files")

    # 保存结果信息
    info_path = output_dir / "convert_info.json"
    with open(info_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
