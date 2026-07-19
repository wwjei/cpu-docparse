"""
DocParser: 极简 CPU 文档解析核心模块

链路: 版面检测 → 全页OCR → 表格结构(ONNX Runtime) → Markdown

推理后端自动选择 (backend.py):
- Intel x86_64 → OpenVINO (AMX/VNNI 加速)
- AMD x86_64 / ARM64 / 其他 → ONNX Runtime
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from cpu_docparse.backend import (
    Backend,
    get_backend_info,
    select_backend,
)

# 默认模型路径 (相对于项目根目录)
_DEFAULT_MODELS_DIR = Path(__file__).parent.parent / "models"

# RapidOCR 配置: 按后端区分
_OCR_CONFIGS = {
    Backend.OPENVINO: Path(__file__).parent.parent / "rapidocr_openvino.yaml",
    Backend.ONNXRUNTIME: Path(__file__).parent.parent / "rapidocr_onnxruntime.yaml",
}

# SLANet 字符字典 (来自 inference.yml, merge_no_span_structure=true)
_SLANET_DICT_RAW = [
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
_SLANET_DICT_MERGED = [x for x in _SLANET_DICT_RAW if x != '<td>'] + ['<td></td>']
_SLANET_CHARS = ['sos'] + _SLANET_DICT_MERGED + ['eos']
_TD_TOKENS = {'<td>', '<td', '<td></td>'}

# 版面检测标签 (PP-DocLayoutV3, 25 类)
_LAYOUT_LABELS = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer",
    "footer_image", "footnote", "formula_number", "header",
    "header_image", "image", "inline_formula", "number",
    "paragraph_title", "reference", "reference_content", "seal",
    "table", "text", "vertical_text", "vision_footnote"
]


class DocParser:
    """极简 CPU 文档解析器

    将扫描件/图片解析为结构化 Markdown。全 CPU 推理，无需 GPU。

    推理后端自动选择：Intel CPU 用 OpenVINO，AMD/ARM/其他用 ONNX Runtime。

    Args:
        models_dir: 模型文件目录，默认 ./models/
        ocr_config: RapidOCR 配置文件路径，默认按后端自动选择
        layout_threshold: 版面检测置信度阈值，默认 0.5
        backend: 推理后端 "auto" / "openvino" / "onnxruntime"，默认 auto
        verbose: 是否打印初始化信息

    Example:
        >>> from cpu_docparse import DocParser
        >>> parser = DocParser()
        >>> result = parser.parse("scan.png")
        >>> print(result["markdown"])
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        ocr_config: Optional[str] = None,
        layout_threshold: float = 0.5,
        backend: str = "auto",
        verbose: bool = True,
    ):
        from rapidocr import RapidOCR
        import onnxruntime as ort

        self._models_dir = Path(models_dir) if models_dir else _DEFAULT_MODELS_DIR
        self._layout_threshold = layout_threshold
        self._verbose = verbose

        # 自动选择推理后端
        self._backend = select_backend(backend if backend != "auto" else None)
        self._backend_info = get_backend_info(self._backend)

        # OCR 配置: 用户指定 > 按后端默认
        if ocr_config:
            self._ocr_config = str(ocr_config)
        else:
            self._ocr_config = str(_OCR_CONFIGS[self._backend])

        self._log("初始化 DocParser...")
        self._log(f"  推理后端: {self._backend.value} "
                  f"(CPU: {self._backend_info['cpu_vendor']}/{self._backend_info['arch']})")

        # 版面检测
        t0 = time.perf_counter()
        layout_path = str(self._models_dir / "PP-DocLayoutV3.onnx")
        if self._backend == Backend.OPENVINO:
            from openvino import Core
            core = Core()
            self._layout_model = core.compile_model(layout_path, "CPU")
            self._layout_engine = "openvino"
        else:
            self._layout_model = ort.InferenceSession(
                layout_path, providers=["CPUExecutionProvider"]
            )
            self._layout_engine = "onnxruntime"
        self._log(f"  版面检测 ({self._layout_engine}): {time.perf_counter()-t0:.2f}s")

        # OCR: RapidOCR (后端由配置文件决定)
        t0 = time.perf_counter()
        self._ocr = RapidOCR(config_path=self._ocr_config)
        self._log(f"  OCR (RapidOCR/{self._backend.value}): {time.perf_counter()-t0:.2f}s")

        # 表格结构识别: SLANet (ONNX Runtime, 跨平台统一)
        t0 = time.perf_counter()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        slanet_path = str(self._models_dir / "SLANet_fixed.onnx")
        self._slanet = ort.InferenceSession(
            slanet_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._log(f"  表格结构 (SLANet/ONNX Runtime): {time.perf_counter()-t0:.2f}s")

        self._log("初始化完成\n")

    def parse(self, image_path: str) -> dict:
        """解析单页文档图片

        Args:
            image_path: 图片路径 (PNG/JPG/TIFF 等)

        Returns:
            dict with keys:
                - markdown: str, 结构化 Markdown 输出
                - regions: list[dict], 版面区域 (label/bbox/score/order/content)
                - timings: dict, 各阶段耗时(秒)
                - total_time: float, 总耗时(秒)
        """
        image_path = str(image_path)
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        timings = {}

        # Step 1: 版面检测
        t0 = time.perf_counter()
        regions = self._detect_layout(img_np)
        timings["layout"] = time.perf_counter() - t0

        # Step 2: 全页 OCR
        t0 = time.perf_counter()
        ocr_lines = self._ocr_full_page(image_path)
        timings["ocr"] = time.perf_counter() - t0

        # Step 3: 坐标分配
        t0 = time.perf_counter()
        self._assign_ocr_to_regions(regions, ocr_lines)
        timings["assign"] = time.perf_counter() - t0

        # Step 4: 表格结构识别
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

    # ─── 内部方法 ───────────────────────────────────────────────────────────

    def _log(self, msg: str):
        if self._verbose:
            print(f"[DocParser] {msg}")

    def _detect_layout(self, img_np: np.ndarray) -> list:
        """版面检测 (PP-DocLayoutV3, 后端自动)"""
        h, w = img_np.shape[:2]
        input_size = 800

        img_resized = cv2.resize(img_np, (input_size, input_size))
        img_input = img_resized.astype(np.float32) / 255.0
        img_input = img_input.transpose(2, 0, 1)[np.newaxis, ...]

        im_shape = np.array([[input_size, input_size]], dtype=np.float32)
        scale_factor = np.array([[input_size / h, input_size / w]], dtype=np.float32)

        if self._layout_engine == "openvino":
            result = self._layout_model({
                "im_shape": im_shape,
                "image": img_input,
                "scale_factor": scale_factor,
            })
            det_output = list(result.values())[0]
        else:
            det_output = self._layout_model.run(
                None,
                {
                    "im_shape": im_shape,
                    "image": img_input,
                    "scale_factor": scale_factor,
                },
            )[0]

        regions = []
        for det in det_output:
            cls_id, score, x1, y1, x2, y2 = det[0], det[1], det[2], det[3], det[4], det[5]
            if score < self._layout_threshold:
                continue
            label = _LAYOUT_LABELS[int(cls_id)] if int(cls_id) < len(_LAYOUT_LABELS) else f"class_{int(cls_id)}"
            regions.append({
                "label": label,
                "score": float(score),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "order": None,
                "content": "",
            })

        # 按 y 坐标排序确定阅读顺序
        regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))
        order_idx = 1
        for r in regions:
            if r["label"] not in ("table", "image", "chart", "display_formula",
                                  "inline_formula", "figure_title"):
                r["order"] = order_idx
                order_idx += 1

        return regions

    def _ocr_full_page(self, image_path: str) -> list:
        """全页 OCR 一次推理 (RapidOCR)"""
        result = self._ocr(image_path)
        lines = []
        if result and result.txts:
            for text, box in zip(result.txts, result.boxes):
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

    def _assign_ocr_to_regions(self, regions: list, ocr_lines: list):
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

    def _build_table_with_slanet(self, region: dict, img_np: np.ndarray, ocr_lines: list):
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

        # SLANet 预处理: resize(保持比例, 最长边488) + pad(488x488) + normalize
        input_size = 488
        ratio = input_size / max(crop_h, crop_w)
        new_w, new_h = int(crop_w * ratio), int(crop_h * ratio)
        resized = cv2.resize(table_crop, (new_w, new_h))
        padded = np.full((input_size, input_size, 3), 127.5, dtype=np.float32)
        padded[:new_h, :new_w, :] = resized.astype(np.float32)
        img_input = padded / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_input = (img_input - mean) / std
        img_input = img_input.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        # 推理
        outputs = self._slanet.run(None, {"x": img_input})
        structure_probs = outputs[1][0]  # [seq, 50]

        # 解码结构 token
        structure_ids = np.argmax(structure_probs, axis=1)
        eos_idx = _SLANET_CHARS.index('eos')
        sos_idx = _SLANET_CHARS.index('sos')

        structure_tokens = []
        for idx in range(len(structure_ids)):
            char_idx = int(structure_ids[idx])
            if idx > 0 and char_idx == eos_idx:
                break
            if char_idx in (sos_idx, eos_idx):
                continue
            structure_tokens.append(_SLANET_CHARS[char_idx])

        # 解析行列数
        rows_cols = self._parse_structure_to_rows(structure_tokens)

        if not rows_cols:
            self._build_table_fallback(region, ocr_lines)
            return

        num_rows = len(rows_cols)
        num_cols = max(rows_cols) if rows_cols else 1

        # 筛选表格区域内的 OCR 行
        table_ocr_lines = [
            line for line in ocr_lines
            if rx1 <= line["center"][0] <= rx2 and ry1 <= line["center"][1] <= ry2
        ]

        if not table_ocr_lines:
            region["content"] = "[表格无文字]"
            return

        # 按 y 分行
        table_ocr_lines.sort(key=lambda l: l["bbox"][1])
        ocr_rows = self._cluster_into_rows(table_ocr_lines, num_rows)

        # 每行内按 x 分列
        grid = []
        for row_lines in ocr_rows:
            row_lines.sort(key=lambda l: l["bbox"][0])
            cells = [l["text"].strip() for l in row_lines]
            while len(cells) < num_cols:
                cells.append(" ")
            grid.append(cells[:num_cols])

        while len(grid) < num_rows:
            grid.append([" "] * num_cols)

        # 构建 Markdown 表格
        md_lines = []
        for i, row in enumerate(grid):
            md_lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                md_lines.append("|" + "|".join(["---"] * num_cols) + "|")

        region["content"] = "\n".join(md_lines)

    def _cluster_into_rows(self, lines: list, expected_rows: int) -> list:
        """将 OCR 行按 y 坐标聚类为 expected_rows 行"""
        if not lines:
            return []
        if len(lines) <= expected_rows:
            return [[l] for l in lines]

        lines.sort(key=lambda l: l["bbox"][1])
        gaps = []
        for i in range(1, len(lines)):
            gap = lines[i]["bbox"][1] - lines[i - 1]["bbox"][1]
            gaps.append((gap, i))

        gaps.sort(reverse=True)
        split_indices = sorted([g[1] for g in gaps[:expected_rows - 1]])

        rows = []
        prev = 0
        for idx in split_indices:
            rows.append(lines[prev:idx])
            prev = idx
        rows.append(lines[prev:])
        return rows

    def _parse_structure_to_rows(self, tokens: list) -> list:
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
            elif token in _TD_TOKENS:
                current_row_cols += 1

        return rows

    def _build_table_fallback(self, region: dict, ocr_lines: list):
        """表格 fallback: 纯坐标聚类 (SLANet 失败时使用)"""
        rx1, ry1, rx2, ry2 = region["bbox"]

        table_lines = [
            line for line in ocr_lines
            if rx1 <= line["center"][0] <= rx2 and ry1 <= line["center"][1] <= ry2
        ]

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

    def _build_markdown(self, regions: list) -> str:
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
