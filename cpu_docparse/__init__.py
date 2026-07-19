"""
cpu-docparse: 极简 CPU 文档解析库

将扫描件/图片解析为结构化 Markdown，全 CPU 推理，无需 GPU。
链路: 版面检测 → 阅读顺序 → 表格结构识别 → 文字 OCR → Markdown 输出

Usage:
    from cpu_docparse import DocParser

    parser = DocParser()
    result = parser.parse("scan.png")
    print(result["markdown"])
"""

from cpu_docparse.parser import DocParser

__version__ = "0.1.0"
__all__ = ["DocParser"]
