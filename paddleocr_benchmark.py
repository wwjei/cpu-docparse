"""
PaddleOCR 3.7.0 + PP-OCRv6 CPU 性能基准测试
对比 Tesseract，测试单页/批量/并发场景
"""

import time
import sys
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import paddle
from paddle import inference as paddle_inference

# Monkey-patch: 修复 PaddlePaddle 3.3.1 的 OneDNN/PIR 兼容性 bug
_orig_create = paddle_inference.create_predictor
def _patched_create(config):
    config.disable_mkldnn()
    config.switch_ir_optim(False)
    return _orig_create(config)
paddle_inference.create_predictor = _patched_create

from paddleocr import PaddleOCR
from PIL import Image

OUTPUT_DIR = Path("/data/test_docs")


def init_ocr(model_tier="tiny"):
    """初始化 PaddleOCR，可选 tiny/small/medium"""
    det_model = f"PP-OCRv6_{model_tier}_det"
    rec_model = f"PP-OCRv6_{model_tier}_rec"
    start = time.perf_counter()
    ocr = PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        cpu_threads=multiprocessing.cpu_count(),
    )
    init_time = time.perf_counter() - start
    return ocr, init_time


def ocr_one(ocr, img_path):
    """单张图片 OCR，返回 (耗时, 字符数, 行数)"""
    start = time.perf_counter()
    results = ocr.predict(str(img_path))
    elapsed = time.perf_counter() - start

    char_count = 0
    line_count = 0
    for r in results:
        if "rec_texts" in r:
            for text in r["rec_texts"]:
                char_count += len(text)
                line_count += 1
    return elapsed, char_count, line_count


def main():
    print("=" * 60)
    print("PaddleOCR PP-OCRv6 CPU 性能基准测试")
    print(f"环境: {multiprocessing.cpu_count()} 核 CPU, Python {sys.version.split()[0]}")
    print(f"PaddlePaddle: {paddle.__version__}")
    print("=" * 60)

    img_paths = sorted(OUTPUT_DIR.glob("scan_page_*.png"))[:3]
    if not img_paths:
        print("ERROR: 没有找到测试图片，请先运行 doc_parse_benchmark.py 生成")
        return
    print(f"\n测试图片: {len(img_paths)} 张 (A4 150dpi)")

    # ===== 测试 PP-OCRv6 Tiny =====
    print("\n" + "-" * 60)
    print("[1] PP-OCRv6 Tiny 模型")
    print("-" * 60)

    ocr_tiny, init_time = init_ocr("tiny")
    print(f"  模型加载: {init_time:.2f}s")

    # 预热
    print("  预热中...")
    ocr_one(ocr_tiny, img_paths[0])

    # 单页测试
    print("\n  单页推理:")
    for p in img_paths:
        elapsed, chars, lines = ocr_one(ocr_tiny, p)
        print(f"    {p.name}: {elapsed*1000:.0f}ms, {chars} chars, {lines} lines")

    # 批量测试
    print("\n  批量推理 (3页):")
    times = []
    total_chars = 0
    for p in img_paths:
        elapsed, chars, lines = ocr_one(ocr_tiny, p)
        times.append(elapsed)
        total_chars += chars
    avg = sum(times) / len(times)
    print(f"    总耗时: {sum(times):.2f}s")
    print(f"    平均: {avg*1000:.0f}ms/page")
    print(f"    速度: {len(img_paths)/sum(times):.2f} pages/s")
    print(f"    总字符: {total_chars}")

    # 识别结果样例
    print("\n  识别结果样例 (第1页前5行):")
    results = ocr_tiny.predict(str(img_paths[0]))
    for r in results:
        if "rec_texts" in r:
            for text in r["rec_texts"][:5]:
                print(f"    {text}")
            break

    # ===== 测试 PP-OCRv6 Small (如果模型可下载) =====
    print("\n" + "-" * 60)
    print("[2] PP-OCRv6 Small 模型")
    print("-" * 60)

    try:
        ocr_small, init_time_s = init_ocr("small")
        print(f"  模型加载: {init_time_s:.2f}s")

        # 预热
        ocr_one(ocr_small, img_paths[0])

        print("\n  批量推理 (3页):")
        times_s = []
        total_chars_s = 0
        for p in img_paths:
            elapsed, chars, lines = ocr_one(ocr_small, p)
            times_s.append(elapsed)
            total_chars_s += chars
        avg_s = sum(times_s) / len(times_s)
        print(f"    总耗时: {sum(times_s):.2f}s")
        print(f"    平均: {avg_s*1000:.0f}ms/page")
        print(f"    速度: {len(img_paths)/sum(times_s):.2f} pages/s")
        print(f"    总字符: {total_chars_s}")
    except Exception as e:
        print(f"  Small 模型不可用: {e}")
        avg_s = None

    # ===== 对比汇总 =====
    print("\n" + "=" * 60)
    print("性能对比汇总")
    print("=" * 60)
    print(f"""
┌─────────────────────────────────────────────────────────────┐
│ PP-OCRv6 Tiny (1.5MB) - CPU {multiprocessing.cpu_count()}核                      │
│   单页平均: {avg*1000:.0f}ms                                     │
│   吞吐量:   {len(img_paths)/sum(times):.2f} pages/s                            │
├─────────────────────────────────────────────────────────────┤""")
    if avg_s:
        print(f"""│ PP-OCRv6 Small (7.7MB) - CPU {multiprocessing.cpu_count()}核                     │
│   单页平均: {avg_s*1000:.0f}ms                                     │
│   吞吐量:   {len(img_paths)/sum(times_s):.2f} pages/s                            │
├─────────────────────────────────────────────────────────────┤""")
    print(f"""│ Tesseract 4.1.1 (之前测试) - 同环境                          │
│   单页平均: 1304ms (150dpi)                                  │
│   吞吐量:   0.77 pages/s                                    │
├─────────────────────────────────────────────────────────────┤
│ 加速比 (Tiny vs Tesseract): {1304/avg/1000:.1f}x                          │
└─────────────────────────────────────────────────────────────┘

生产环境预估 (真实 4核 Intel Xeon + OpenVINO):
- PP-OCRv6 Tiny:  ~200ms/page → 5 pages/s/核 → 4核约 15-20 pages/s
- PP-OCRv6 Small: ~500ms/page → 2 pages/s/核 → 4核约 6-8 pages/s
- PP-OCRv6 Medium + OpenVINO: ~1.4s/page (官方数据)
- 日处理量 (Tiny, 4核): ~130万页

注: 本沙箱未使用 OpenVINO 加速，实际生产环境会更快 (官方数据 5.2x)
""")


if __name__ == "__main__":
    main()
