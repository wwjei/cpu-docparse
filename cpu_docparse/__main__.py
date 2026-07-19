"""
cpu-docparse CLI 入口

Usage:
    python -m cpu_docparse scan.png
    python -m cpu_docparse scan.png -o output.md
    python -m cpu_docparse scan.png --json
"""

import argparse
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        prog="cpu-docparse",
        description="极简 CPU 文档解析：扫描件/图片 → 结构化 Markdown",
    )
    parser.add_argument("image", help="输入图片路径 (PNG/JPG/TIFF)")
    parser.add_argument("-o", "--output", help="输出 Markdown 文件路径 (默认打印到 stdout)")
    parser.add_argument("--json", action="store_true", help="输出完整 JSON (含 regions/timings)")
    parser.add_argument("--models-dir", help="模型目录路径 (默认 ./models/)")
    parser.add_argument("--ocr-config", help="RapidOCR 配置文件路径")
    parser.add_argument("--threshold", type=float, default=0.5, help="版面检测置信度阈值 (默认 0.5)")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式，不打印初始化信息")

    args = parser.parse_args()

    from cpu_docparse import DocParser

    doc_parser = DocParser(
        models_dir=args.models_dir,
        ocr_config=args.ocr_config,
        layout_threshold=args.threshold,
        verbose=not args.quiet,
    )

    result = doc_parser.parse(args.image)

    if args.json:
        output = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        output = result["markdown"]

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        if not args.quiet:
            print(f"[DocParser] 已写入 {args.output} ({result['total_time']*1000:.0f}ms)", file=sys.stderr)
    else:
        print(output)

    if not args.quiet and not args.json:
        print(f"\n--- 耗时: {result['total_time']*1000:.0f}ms ---", file=sys.stderr)


if __name__ == "__main__":
    main()
