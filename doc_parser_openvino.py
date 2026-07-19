"""
OpenVINO 加速版文档解析 Pipeline v3 (含 SLANet 表格结构识别)
优化策略:
1. 版面检测: OpenVINO (PP-DocLayoutV3 ONNX)
2. OCR: RapidOCR (OpenVINO 后端, PP-OCRv6 Small) - 全页一次推理
3. 表格: SLANet 结构识别 (ONNX Runtime) + OCR文字填充
4. 架构: 全页 OCR 一次 → 按坐标分配到版面区域
"""

import time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from openvino import Core
from rapidocr import RapidOCR
import onnxruntime as ort


# SLANet 字符字典 (来自 inference.yml, merge_no_span_structure=true)
# 原始: <thead> </thead> <tbody> </tbody> <tr> </tr> <td> <td '>' </td> colspan... rowspan...
# merge后: 去掉 <td>, 末尾加 <td></td>
SLANET_DICT_RAW = [
    '<thead>', '</thead>', '<tbody>', '</tbody>', '<tr>', '</tr>',
    '<td>', '<td', '>', '</td>',
    ' colspan="2"', ' colspan="3"', ' colspan="4"', ' colspan="5"',
    ' colspan="6"', ' colspan="7"', ' colspan="8"', ' colspan="9"',
    ' colspan="10"', ' colspan="11"', ' colspan="12"', ' colspan="13"',
    ' colspan="14"', ' colspan="15"', ' colspan="16"', ' colspan="17"',
    ' colspan="18"', ' colspan="19"', ' colspan="20"',
    ' rowspan="2"', ' rowspan="3"', ' rowspan="4"', ' rowspan="5"',
    ' rowspan="6"', ' rowspan="7"', ' rowspan="8"', ' rowspan="9"',
    ' rowspan="10"', ' rowspan="11"', ' rowspan="12"', ' rowspan="13"',
    ' rowspan="14"', ' rowspan="15"', ' rowspan="16"', ' rowspan="17"',
    ' rowspan="18"', ' rowspan="19"', ' rowspan="20"',
]
# merge_no_span_structure: remove <td>, append <td></td>
SLANET_DICT_MERGED = [x for x in SLANET_DICT_RAW if x != '<td>'] + ['<td></td>']
# add sos/eos
SLANET_CHARS = ['sos'] + SLANET_DICT_MERGED + ['eos']
TD_TOKENS = {'<td>', '<td', '<td></td>'}


class FastDocParser:
    """OpenVINO + RapidOCR + SLANet 文档解析器"""

    def __init__(self):
        print("[FastDocParser] 初始化...")
        self.core = Core()

        # 版面检测: OpenVINO
        t0 = time.perf_counter()
        self.layout_model = self.core.compile_model(
            "/data/models/PP-DocLayoutV3.onnx", "CPU"
        )
        print(f"  版面检测 (OpenVINO): {time.perf_counter()-t0:.2f}s")

        # OCR: RapidOCR (OpenVINO 后端, PP-OCRv6 Small)
        t0 = time.perf_counter()
        self.ocr = RapidOCR(config_path="/data/rapidocr_openvino.yaml")
        print(f"  OCR (RapidOCR/OpenVINO): {time.perf_counter()-t0:.2f}s")

        # 表格结构识别: SLANet (ONNX Runtime, 修复后的模型)
        t0 = time.perf_counter()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.slanet = ort.InferenceSession(
            "/data/models/SLANet_fixed.onnx", sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        print(f"  表格结构 (SLANet/ONNX Runtime): {time.perf_counter()-t0:.2f}s")

        # 标签映射 (来自 inference.yml)
        self.labels = [
            "abstract", "algorithm", "aside_text", "chart", "content",
            "display_formula", "doc_title", "figure_title", "footer",
            "footer_image", "footnote", "formula_number", "header",
            "header_image", "image", "inline_formula", "number",
            "paragraph_title", "reference", "reference_content", "seal",
            "table", "text", "vertical_text", "vision_footnote"
        ]
        print("[FastDocParser] 初始化完成\n")

    def parse(self, image_path) -> dict:
        """解析单页文档"""
        image_path = str(image_path)
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        timings = {}

        # Step 1: 版面检测 (OpenVINO)
        t0 = time.perf_counter()
        regions = self._detect_layout(img_np)
        timings["layout"] = time.perf_counter() - t0

        # Step 2: 全页 OCR 一次推理 (RapidOCR)
        t0 = time.perf_counter()
        ocr_lines = self._ocr_full_page(image_path)
        timings["ocr"] = time.perf_counter() - t0

        # Step 3: 将 OCR 结果按坐标分配到版面区域
        t0 = time.perf_counter()
        self._assign_ocr_to_regions(regions, ocr_lines)
        timings["assign"] = time.perf_counter() - t0

        # Step 4: 表格结构识别 (SLANet) + OCR文字填充
        t0 = time.perf_counter()
        for region in regions:
            if region["label"] == "table":
                self._build_table_with_slanet(region, img_np, ocr_lines)
        timings["table"] = time.perf_counter() - t0

        # Step 5: 组装 Markdown
        t0 = time.perf_counter()
        markdown = self._build_markdown(regions)
        timings["markdown"] = time.perf_counter() - t0

        return {
            "input": image_path,
            "regions": regions,
            "markdown": markdown,
            "timings": timings,
            "total_time": sum(timings.values()),
        }

    def _detect_layout(self, img_np):
        """版面检测 (OpenVINO, PP-DocLayoutV3)"""
        h, w = img_np.shape[:2]
        input_size = 800

        img_resized = cv2.resize(img_np, (input_size, input_size))
        img_input = img_resized.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        im_shape = np.array([[input_size, input_size]], dtype=np.float32)
        scale_factor = np.array([[input_size / h, input_size / w]], dtype=np.float32)

        result = self.layout_model({
            "im_shape": im_shape,
            "image": img_input,
            "scale_factor": scale_factor,
        })

        outputs = list(result.values())
        det_output = outputs[0]

        regions = []
        for i in range(len(det_output)):
            det = det_output[i]
            cls_id, score, x1, y1, x2, y2 = det[0], det[1], det[2], det[3], det[4], det[5]
            if score < 0.5:
                continue
            label = self.labels[int(cls_id)] if int(cls_id) < len(self.labels) else f"class_{int(cls_id)}"
            regions.append({
                "label": label,
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "order": None,
                "content": "",
            })

        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        order_idx = 1
        for r in regions:
            if r["label"] not in ("table", "image", "chart", "display_formula",
                                  "inline_formula", "figure_title"):
                r["order"] = order_idx
                order_idx += 1

        return regions

    def _ocr_full_page(self, image_path):
        """全页 OCR 一次推理 (RapidOCR)"""
        result = self.ocr(image_path)
        lines = []
        if result and result.txts:
            for i, (text, box) in enumerate(zip(result.txts, result.boxes)):
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                lines.append({
                    "text": text,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "center": [(x_min + x_max) / 2, (y_min + y_max) / 2],
                })
        return lines

    def _assign_ocr_to_regions(self, regions, ocr_lines):
        """将 OCR 行按坐标中心点分配到版面区域"""
        for region in regions:
            if region["label"] in ("table", "image", "figure", "formula"):
                continue
            rx1, ry1, rx2, ry2 = region["bbox"]
            pad = 5
            rx1, ry1 = rx1 - pad, ry1 - pad
            rx2, ry2 = rx2 + pad, ry2 + pad

            matched = []
            for line in ocr_lines:
                cx, cy = line["center"]
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    matched.append(line)

            matched.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
            region["content"] = " ".join(l["text"] for l in matched)

    def _build_table_with_slanet(self, region, img_np, ocr_lines):
        """表格区域: SLANet 结构识别 (行列数) + 坐标聚类填充文字"""
        rx1, ry1, rx2, ry2 = [int(v) for v in region["bbox"]]
        h_img, w_img = img_np.shape[:2]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w_img, rx2), min(h_img, ry2)

        table_crop = img_np[ry1:ry2, rx1:rx2]
        if table_crop.size == 0:
            region["content"] = "[表格区域为空]"
            return

        crop_h, crop_w = table_crop.shape[:2]

        # SLANet 预处理: ResizeTableImage(max_len=488) + PaddingTableImage(488x488) + Normalize
        input_size = 488
        ratio = input_size / max(crop_h, crop_w)
        new_w = int(crop_w * ratio)
        new_h = int(crop_h * ratio)
        resized = cv2.resize(table_crop, (new_w, new_h))
        padded = np.full((input_size, input_size, 3), 127.5, dtype=np.float32)
        padded[:new_h, :new_w, :] = resized.astype(np.float32)
        img_input = padded / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_input = (img_input - mean) / std
        img_input = img_input.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # 推理
        outputs = self.slanet.run(None, {"x": img_input})
        structure_probs = outputs[1][0]  # [seq, 50]

        # 解码结构 token → 获取行列数
        structure_ids = np.argmax(structure_probs, axis=1)
        eos_idx = SLANET_CHARS.index('eos')
        sos_idx = SLANET_CHARS.index('sos')

        structure_tokens = []
        for idx in range(len(structure_ids)):
            char_idx = int(structure_ids[idx])
            if idx > 0 and char_idx == eos_idx:
                break
            if char_idx == sos_idx or char_idx == eos_idx:
                continue
            token = SLANET_CHARS[char_idx]
            structure_tokens.append(token)

        # 从结构 token 解析行列数
        rows_cols = self._parse_structure_to_rows(structure_tokens)

        if not rows_cols:
            self._build_table_fallback(region, ocr_lines)
            return

        num_rows = len(rows_cols)
        num_cols = max(rows_cols) if rows_cols else 1

        # 用坐标聚类将 OCR 文字分配到 num_rows × num_cols 网格
        table_ocr_lines = []
        for line in ocr_lines:
            cx, cy = line["center"]
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                table_ocr_lines.append(line)

        if not table_ocr_lines:
            region["content"] = "[表格无文字]"
            return

        # 按 y 分行 (使用 SLANet 给出的行数作为参考)
        table_ocr_lines.sort(key=lambda l: l["bbox"][1])
        ocr_rows = self._cluster_into_rows(table_ocr_lines, num_rows)

        # 每行内按 x 分列
        grid = []
        for row_lines in ocr_rows:
            row_lines.sort(key=lambda l: l["bbox"][0])
            cells = [l["text"].strip() for l in row_lines]
            # 补齐到 num_cols
            while len(cells) < num_cols:
                cells.append(" ")
            grid.append(cells[:num_cols])

        # 补齐到 num_rows
        while len(grid) < num_rows:
            grid.append([" "] * num_cols)

        # 构建 Markdown 表格
        md_lines = []
        for i, row in enumerate(grid):
            md_lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                md_lines.append("|" + "|".join(["---"] * num_cols) + "|")

        region["content"] = "\n".join(md_lines)

    def _cluster_into_rows(self, lines, expected_rows):
        """将 OCR 行按 y 坐标聚类为 expected_rows 行"""
        if not lines:
            return []
        if len(lines) <= expected_rows:
            return [[l] for l in lines]

        # 按 y 排序后均匀分割
        lines.sort(key=lambda l: l["bbox"][1])

        # 使用 y 间隙聚类
        gaps = []
        for i in range(1, len(lines)):
            gap = lines[i]["bbox"][1] - lines[i-1]["bbox"][1]
            gaps.append((gap, i))

        # 取最大的 (expected_rows - 1) 个间隙作为分割点
        gaps.sort(reverse=True)
        split_indices = sorted([g[1] for g in gaps[:expected_rows - 1]])

        rows = []
        prev = 0
        for idx in split_indices:
            rows.append(lines[prev:idx])
            prev = idx
        rows.append(lines[prev:])

        return rows

    def _parse_structure_to_rows(self, tokens):
        """解析结构 token 序列 → 每行的列数列表"""
        rows = []
        current_row_cols = 0
        in_row = False

        for token in tokens:
            if token == '<tr>':
                in_row = True
                current_row_cols = 0
            elif token == '</tr>':
                if in_row:
                    rows.append(current_row_cols)
                in_row = False
            elif token in TD_TOKENS:
                current_row_cols += 1

        return rows

    def _build_table_fallback(self, region, ocr_lines):
        """表格 fallback: 坐标聚类 (SLANet 失败时使用)"""
        rx1, ry1, rx2, ry2 = region["bbox"]

        table_lines = []
        for line in ocr_lines:
            cx, cy = line["center"]
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                table_lines.append(line)

        if not table_lines:
            region["content"] = "[表格无文字]"
            return

        table_lines.sort(key=lambda l: l["bbox"][1])
        rows = []
        current_row = [table_lines[0]]
        for i in range(1, len(table_lines)):
            y_gap = table_lines[i]["bbox"][1] - current_row[-1]["bbox"][1]
            row_height = current_row[-1]["bbox"][3] - current_row[-1]["bbox"][1]
            if y_gap > max(row_height * 0.5, 10):
                rows.append(current_row)
                current_row = [table_lines[i]]
            else:
                current_row.append(table_lines[i])
        rows.append(current_row)

        for row in rows:
            row.sort(key=lambda l: l["bbox"][0])

        max_cols = max(len(row) for row in rows)
        md_lines = []
        for i, row in enumerate(rows):
            cells = [l["text"].strip() for l in row]
            while len(cells) < max_cols:
                cells.append(" ")
            md_lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_lines.append("|" + "|".join(["---"] * max_cols) + "|")

        region["content"] = "\n".join(md_lines)

    def _build_markdown(self, regions):
        """按阅读顺序组装 Markdown"""
        ordered = [r for r in regions if r.get("order") is not None]
        tables = [r for r in regions if r["label"] == "table" and r.get("content")]

        ordered.sort(key=lambda r: r["order"])

        md_parts = []
        table_inserted = set()

        for region in ordered:
            for t in tables:
                if id(t) not in table_inserted and t["bbox"][1] < region["bbox"][1]:
                    md_parts.append("\n" + t["content"] + "\n")
                    table_inserted.add(id(t))

            content = region.get("content", "").strip()
            if not content:
                continue

            label = region["label"]
            if label == "doc_title":
                md_parts.append(f"# {content}")
            elif label == "paragraph_title":
                md_parts.append(f"## {content}")
            else:
                md_parts.append(content)

        for t in tables:
            if id(t) not in table_inserted:
                md_parts.append("\n" + t["content"] + "\n")

        return "\n\n".join(md_parts)


def main():
    print("=" * 60)
    print("OpenVINO 加速版文档解析 Pipeline v3 (含 SLANet 表格)")
    print("=" * 60)

    parser = FastDocParser()

    test_img = "/data/test_docs/doc_with_table.png"
    print(f"解析文档: {test_img}")
    print("-" * 60)

    # 预热
    print("预热...")
    parser.parse(test_img)

    # 正式测试 (3次)
    print("\n正式测试 (3次):")
    all_timings = []
    result = None
    for i in range(3):
        result = parser.parse(test_img)
        all_timings.append(result["timings"])
        print(f"  Round {i+1}: total {result['total_time']*1000:.0f}ms")

    # 平均耗时
    avg_timings = {}
    for key in all_timings[0]:
        avg_timings[key] = sum(t[key] for t in all_timings) / len(all_timings)

    total = sum(avg_timings.values())
    print(f"\n{'='*60}")
    print("耗时拆分 (3次平均):")
    print(f"{'='*60}")
    for key, val in avg_timings.items():
        pct = val / total * 100
        print(f"  {key:<12}: {val*1000:>7.0f}ms  ({pct:.0f}%)")
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<12}: {total*1000:>7.0f}ms")
    print(f"\n  速度: {1/total:.2f} pages/s")

    print(f"\n{'='*60}")
    print("Markdown 输出:")
    print("=" * 60)
    print(result["markdown"])
    print("=" * 60)

    # 区域详情
    print(f"\n区域详情:")
    for r in result["regions"]:
        order_str = f"order={r['order']}" if r.get('order') is not None else "order=None"
        content_preview = r.get("content", "")[:60].replace("\n", " ")
        print(f"  [{r['label']:<16}] {order_str:<12} conf={r['score']:.3f}  {content_preview}")

    # 对比
    print(f"""
{'='*60}
性能对比:
{'='*60}
┌───────────────────────────────────────────────────────────────────────┐
│ 方案                                │ 耗时      │ 速度      │ 加速比  │
├───────────────────────────────────────────────────────────────────────┤
│ v0 裸 Paddle 逐区域 OCR             │ 10,380ms  │ 0.10 p/s  │ 1.0x   │
│ v2 OpenVINO + 坐标聚类表格          │  1,139ms  │ 0.88 p/s  │ 9.1x   │
│ v3 OpenVINO + SLANet表格结构识别    │ {total*1000:>6.0f}ms  │ {1/total:.2f} p/s  │ {10380/total/1000:.1f}x   │
└───────────────────────────────────────────────────────────────────────┘

各阶段:
  版面检测 (PP-DocLayoutV3/OpenVINO):  {avg_timings['layout']*1000:.0f}ms
  全页OCR (PP-OCRv6 Small/OpenVINO):   {avg_timings['ocr']*1000:.0f}ms
  表格结构 (SLANet/ONNX Runtime):      {avg_timings['table']*1000:.0f}ms
  坐标分配+Markdown组装:               {(avg_timings['assign']+avg_timings['markdown'])*1000:.0f}ms
""")


if __name__ == "__main__":
    main()
