# CPU Doc-Parse — 私有化文档解析服务

CPU 上运行的企业级文档解析 pipeline：版面检测 → 阅读顺序 → 表格解析 → 文字 OCR，
输出结构化 Markdown / JSON，供 LLM 消费。不依赖 GPU，全 CPU 推理。

## 最终方案性能 (v3)

单页 A4 150dpi 文档，沙箱受限约 1 核实测：

| 阶段 | 模型 | 推理后端 | 耗时 | 占比 |
|------|------|---------|------|------|
| 版面检测 | PP-DocLayoutV3 (124.5MB) | OpenVINO | 722ms | 55% |
| 全页 OCR | PP-OCRv6 Small det+rec (30MB) | OpenVINO | 527ms | 40% |
| 表格结构识别 | SLANet (7.4MB) | ONNX Runtime | 74ms | 6% |
| 坐标分配 + Markdown 组装 | 纯算法 | — | <1ms | ~0% |
| **总计** | | | **1,323ms** | **0.76 pages/s** |

相比裸 Paddle 逐区域方案 (10,380ms) 加速 **7.8 倍**。

## 架构

```
输入图片 / 扫描件
  ├─ 版面检测  PP-DocLayoutV3 (OpenVINO)   → 25 类区域 + bbox + 阅读顺序
  ├─ 全页 OCR  PP-OCRv6 Small (OpenVINO)    → 文字行 + 坐标 (一次推理，不逐区域裁剪)
  ├─ 表格结构  SLANet (ONNX Runtime)        → 行列结构 (支持合并单元格 / 无线表格)
  ├─ 坐标分配  纯算法                        → OCR 行按中心点归入版面区域
  └─ Markdown  纯算法                        → 按阅读顺序拼接标题 / 正文 / 表格
```

关键优化：全页 OCR 只推理一次，再按坐标把文字分配到各版面区域，避免逐区域裁剪 N 次推理。

## 文件说明

| 文件 | 说明 |
|------|------|
| `doc_parser_openvino.py` | **最终方案 v3** — OpenVINO + SLANet 全链路 pipeline |
| `doc_parser.py` | v0 裸 Paddle 版 (对照基线) |
| `doc_parse_benchmark.py` | Tesseract + PyMuPDF 基准测试 |
| `paddleocr_benchmark.py` | PaddleOCR PP-OCRv6 裸推理测试 |
| `rapidocr_openvino.yaml` | RapidOCR OpenVINO 后端配置 |
| `download_models.sh` | 下载并转换版面检测模型 (见下) |
| `phase.md` | 完整测试记录 (Phase 1–7) |
| `文档解析服务_技术报告.md` | 技术报告 + 多架构加速框架选型 |
| `models/SLANet_fixed.onnx` | 表格结构模型 (已修复 Loop 算子) |
| `test_docs/` | 测试文档 |

## 模型文件

`models/PP-DocLayoutV3.onnx` (~125MB) 超过 GitHub 100MB 限制，未提交到仓库。
首次使用前运行下载脚本：

```bash
chmod +x download_models.sh
./download_models.sh
```

该脚本通过 PaddleX 拉取官方 Paddle 推理模型并用 paddle2onnx 转换为 ONNX。
需要 `pip install paddlepaddle paddle2onnx paddlex`。

PP-OCRv6 Small 的 det/rec ONNX 模型随 `rapidocr` 包自带，无需单独下载。

## 依赖

```bash
pip install openvino onnxruntime rapidocr pymupdf opencv-python numpy pillow
# 仅下载/转换模型时需要:
pip install paddlepaddle paddle2onnx paddlex
```

## 运行

```bash
python doc_parser_openvino.py
```

## 多架构部署

OpenVINO 在 Intel CPU 上性能最优，但也支持 ARM64 / AMD (通用 x86 路径)。
跨架构部署建议「ONNX Runtime 通用 + OpenVINO Intel 加速」双后端策略，
详见 `文档解析服务_技术报告.md`。

## 已知问题与修复

- **PaddlePaddle 3.3.1 OneDNN/PIR bug**：改用 OpenVINO 后端绕开。
- **SLANet ONNX Loop 算子无法加载**：在 And 节点输出后插入 Squeeze 节点
  修复 shape 声明，生成 `SLANet_fixed.onnx`，并用 ONNX Runtime (禁用图优化) 加载。
