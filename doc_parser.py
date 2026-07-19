"""
自实现文档解析 Pipeline: 版面检测 + 阅读顺序 + OCR + 表格识别 → Markdown
不依赖 PP-StructureV3 端到端，自己组装各模块
"""

import time
import numpy as np
from pathlib import Path
from PIL import Image

import paddle
from paddle import inference as paddle_inference

# Monkey-patch 修复 PaddlePaddle 3.3.1 OneDNN bug
_orig_create = paddle_inference.create_predictor
def _patched_create(config):
    config.disable_mkldnn()
    config.switch_ir_optim(False)
    return _orig_create(config)
paddle_inference.create_predictor = _patched_create

from paddleocr import LayoutDetection, PaddleOCR, TableStructureRecognition


class DocParser:
    """自组装文档解析器：版面 → 阅读顺序 → OCR → 表格 → Markdown"""

    def __init__(self, device="cpu"):
        print("[DocParser] 初始化模块...")

        t0 = time.perf_counter()
        self.layout = LayoutDetection(model_name="PP-DocLayoutV3", device=device)
        print(f"  版面检测 (PP-DocLayoutV3): {time.perf_counter()-t0:.2f}s")

        t0 = time.perf_counter()
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )
        print(f"  文字 OCR (PP-OCRv6 Small): {time.perf_counter()-t0:.2f}s")

        t0 = time.perf_counter()
        self.table_rec = TableStructureRecognition(device=device)
        print(f"  表格结构 (SLANet): {time.perf_counter()-t0:.2f}s")

        print("[DocParser] 初始化完成\n")

    def parse(self, image_path) -> dict:
        """解析单页文档图片，返回结构化结果"""
        image_path = str(image_path)
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        result = {
            "input": image_path,
            "width": img.width,
            "height": img.height,
            "regions": [],
            "markdown": "",
        }

        # Step 1: 版面检测 + 阅读顺序
        layout_results = list(self.layout.predict(image_path))
        if not layout_results or "boxes" not in layout_results[0]:
            return result

        boxes = layout_results[0]["boxes"]

        # Step 2: 按区域类型分组处理
        text_regions = []
        table_regions = []
        title_regions = []

        for box in boxes:
            label = box["label"]
            coord = box["coordinate"]
            order = box.get("order")
            score = box.get("score", 0)

            region = {
                "label": label,
                "score": float(score),
                "bbox": [float(c) for c in coord[:4]],
                "order": order,
                "content": "",
            }

            if label == "table":
                table_regions.append(region)
            elif label in ("paragraph_title", "doc_title"):
                title_regions.append(region)
            elif label in ("text", "reference", "footnote"):
                text_regions.append(region)
            # figure_title, table_caption 等暂跳过

            result["regions"].append(region)

        # Step 3: OCR 识别所有文本区域
        all_text_regions = title_regions + text_regions
        if all_text_regions:
            self._ocr_regions(img_np, all_text_regions)

        # Step 4: 表格结构识别 + OCR 填充
        for region in table_regions:
            self._parse_table(img, img_np, region)

        # Step 5: 按阅读顺序组装 Markdown
        result["markdown"] = self._build_markdown(result["regions"])

        return result

    def _ocr_regions(self, img_np, regions):
        """对文本区域做 OCR，按 bbox 裁剪后识别"""
        for region in regions:
            x1, y1, x2, y2 = [int(c) for c in region["bbox"]]
            # 扩展一点边距
            pad = 2
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(img_np.shape[1], x2 + pad)
            y2 = min(img_np.shape[0], y2 + pad)

            crop = img_np[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # 保存临时图片给 OCR
            tmp_path = "/data/test_docs/_tmp_crop.png"
            Image.fromarray(crop).save(tmp_path)

            ocr_results = list(self.ocr.predict(tmp_path))
            if ocr_results and "rec_texts" in ocr_results[0]:
                texts = ocr_results[0]["rec_texts"]
                region["content"] = " ".join(texts)
            else:
                region["content"] = ""

    def _parse_table(self, img_pil, img_np, region):
        """表格区域：结构识别 + OCR 填充单元格"""
        x1, y1, x2, y2 = [int(c) for c in region["bbox"]]
        table_crop = img_pil.crop((x1, y1, x2, y2))
        tmp_table = "/data/test_docs/_tmp_table.png"
        table_crop.save(tmp_table)

        # 表格结构识别
        struct_results = list(self.table_rec.predict(tmp_table))
        if not struct_results:
            region["content"] = "[表格识别失败]"
            return

        sr = struct_results[0]
        html_tokens = sr.get("structure", [])
        cell_bboxes = sr.get("bbox", [])
        html_skeleton = "".join(html_tokens)

        # OCR 识别表格内文字
        ocr_results = list(self.ocr.predict(tmp_table))
        ocr_texts = []
        ocr_boxes = []
        if ocr_results and "rec_texts" in ocr_results[0]:
            ocr_texts = ocr_results[0]["rec_texts"]
            ocr_boxes = ocr_results[0].get("dt_polys", [])

        # 将 OCR 文字匹配到单元格
        cell_contents = self._match_text_to_cells(cell_bboxes, ocr_texts, ocr_boxes)

        # 构建 Markdown 表格
        region["content"] = self._cells_to_markdown_table(html_skeleton, cell_contents)
        region["cell_count"] = len(cell_bboxes)

    def _match_text_to_cells(self, cell_bboxes, ocr_texts, ocr_boxes):
        """将 OCR 识别的文字按位置匹配到对应单元格"""
        cell_contents = [""] * len(cell_bboxes)

        if not ocr_boxes:
            # 如果没有 bbox 信息，按顺序分配
            for i, text in enumerate(ocr_texts):
                if i < len(cell_contents):
                    cell_contents[i] = text
            return cell_contents

        for text, poly in zip(ocr_texts, ocr_boxes):
            if len(poly) < 3:
                continue
            # 文字中心点
            cx = np.mean([p[0] for p in poly[:4]])
            cy = np.mean([p[1] for p in poly[:4]])

            # 找最近的单元格
            best_idx = -1
            best_dist = float("inf")
            for i, cell in enumerate(cell_bboxes):
                if len(cell) < 6:
                    continue
                # 单元格中心
                cell_cx = np.mean(cell[0::2][:4])
                cell_cy = np.mean(cell[1::2][:4])
                dist = (cx - cell_cx) ** 2 + (cy - cell_cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                if cell_contents[best_idx]:
                    cell_contents[best_idx] += " " + text
                else:
                    cell_contents[best_idx] = text

        return cell_contents

    def _cells_to_markdown_table(self, html_skeleton, cell_contents):
        """将 HTML 骨架 + 单元格内容转为 Markdown 表格"""
        # 解析行列数
        rows = html_skeleton.count("<tr>")
        if rows == 0:
            return "[无法解析表格结构]"

        total_cells = len(cell_contents)
        cols = total_cells // rows if rows > 0 else 1
        if cols == 0:
            cols = 1

        # 构建 Markdown 表格
        lines = []
        idx = 0
        for r in range(rows):
            row_cells = []
            for c in range(cols):
                if idx < len(cell_contents):
                    row_cells.append(cell_contents[idx].strip() or " ")
                else:
                    row_cells.append(" ")
                idx += 1
            lines.append("| " + " | ".join(row_cells) + " |")
            if r == 0:
                # 表头分隔线
                lines.append("|" + "|".join(["---"] * cols) + "|")

        return "\n".join(lines)

    def _build_markdown(self, regions):
        """按阅读顺序组装所有区域为 Markdown"""
        # 分离有 order 的和没有的
        ordered = [r for r in regions if r.get("order") is not None]
        unordered = [r for r in regions if r.get("order") is None]

        # 按 order 排序
        ordered.sort(key=lambda r: r["order"])

        md_parts = []

        # 先输出有顺序的文本区域
        for region in ordered:
            content = region.get("content", "").strip()
            if not content:
                continue

            label = region["label"]
            if label == "doc_title":
                md_parts.append(f"# {content}")
            elif label == "paragraph_title":
                md_parts.append(f"## {content}")
            elif label in ("text", "reference", "footnote"):
                md_parts.append(content)
            else:
                md_parts.append(content)

        # 在正确位置插入表格（按 y 坐标判断位置）
        for region in unordered:
            if region["label"] == "table" and region.get("content"):
                # 找表格应该插入的位置（按 y 坐标）
                table_y = region["bbox"][1]
                insert_idx = len(md_parts)
                for i, r in enumerate(ordered):
                    if r["bbox"][1] > table_y:
                        insert_idx = i
                        break
                md_parts.insert(insert_idx, "\n" + region["content"] + "\n")

        return "\n\n".join(md_parts)


def main():
    print("=" * 60)
    print("自实现文档解析 Pipeline 测试")
    print("=" * 60)

    parser = DocParser(device="cpu")

    # 测试
    test_img = "/data/test_docs/doc_with_table.png"
    print(f"解析文档: {test_img}")
    print("-" * 60)

    start = time.perf_counter()
    result = parser.parse(test_img)
    elapsed = time.perf_counter() - start

    print(f"\n解析耗时: {elapsed:.2f}s")
    print(f"检测区域数: {len(result['regions'])}")
    print(f"\n{'='*60}")
    print("Markdown 输出:")
    print("=" * 60)
    print(result["markdown"])
    print("=" * 60)

    # 区域详情
    print(f"\n区域详情:")
    for r in result["regions"]:
        order_str = f"order={r['order']}" if r.get('order') is not None else "order=None"
        content_preview = r.get("content", "")[:60]
        print(f"  [{r['label']:<16}] {order_str:<12} conf={r['score']:.3f}  {content_preview}")


if __name__ == "__main__":
    main()
