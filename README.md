# CPU Doc-Parse — 私有化文档解析服务

在 **CPU** 上运行的企业级文档解析 pipeline。把扫描件 / 图片 / PDF 页面解析成
结构化 **Markdown / JSON**，直接喂给 LLM 使用。**全程不依赖 GPU**，适合私有化部署。

完整链路：**版面检测 → 阅读顺序 → 表格解析 → 文字 OCR → 结构化输出**。

---

## ✨ 特性

- **纯 CPU 推理**：基于 OpenVINO + ONNX Runtime，无需显卡，普通服务器即可部署。
- **完整解析链路**：版面检测（25 类区域）+ 阅读顺序 + 表格结构识别 + 文字 OCR，一次出结构化结果。
- **快**：单页 A4 150dpi 约 **1.3 秒**（沙箱受限单核），相比裸 Paddle 方案加速 **7.8 倍**。
- **表格能力强**：SLANet 表格结构识别，支持**合并单元格、无线表格 / 三线表**。
- **输出即用**：直接产出带标题层级、表格、正确阅读顺序的 Markdown。
- **模型小**：核心模型合计约 160MB，启动快、内存占用低。
- **私有化友好**：所有推理本地完成，文档不出内网。

---

## 🚀 快速开始（Intel CPU）

### 1. 环境要求

- Python 3.10+
- Intel x86_64 CPU（6 代 Core / 1 代 Xeon 及以上，支持 AVX2，有 AVX-512 更佳）
- Linux / Windows / macOS

### 2. 安装依赖

```bash
git clone https://github.com/wwjei/cpu-docparse.git
cd cpu-docparse

# 运行期依赖（推理用）
pip install openvino onnxruntime rapidocr pymupdf opencv-python numpy pillow
```

### 3. 获取版面检测模型

`models/PP-DocLayoutV3.onnx`（约 125MB）超过 GitHub 单文件 100MB 限制，未随仓库提交。
首次使用运行下载脚本（自动拉取官方模型并转换为 ONNX）：

```bash
# 下载/转换模型需要额外的工具链
pip install paddlepaddle paddle2onnx paddlex

chmod +x download_models.sh
./download_models.sh
```

> 其余模型说明：
> - `models/SLANet_fixed.onnx`（表格结构，7.5MB）已随仓库提交，无需下载。
> - PP-OCRv6 Small 的 det / rec 模型随 `rapidocr` 包自带，首次运行自动就绪。

### 4. 运行

```bash
# 跑内置测试文档，输出 Markdown + 各阶段耗时
python doc_parser_openvino.py
```

### 5. 作为库调用

```python
from doc_parser_openvino import FastDocParser

parser = FastDocParser()                 # 初始化一次，加载全部模型
result = parser.parse("test_docs/doc_with_table.png")

print(result["markdown"])                # 结构化 Markdown
print(result["regions"])                 # 版面区域列表 (label/bbox/score/order/content)
print(result["timings"])                 # 各阶段耗时
print(f'{result["total_time"]*1000:.0f}ms')
```

`parse()` 返回字段：

| 字段 | 说明 |
|------|------|
| `markdown` | 结构化 Markdown 文本（标题 / 正文 / 表格，按阅读顺序） |
| `regions` | 版面区域列表，每项含 `label` `bbox` `score` `order` `content` |
| `timings` | 各阶段耗时（秒）：`layout` `ocr` `assign` `table` `markdown` |
| `total_time` | 总耗时（秒） |

---

## 🏗️ 架构

```
输入图片 / 扫描件
  ├─ 版面检测  PP-DocLayoutV3 (OpenVINO)    → 25 类区域 + bbox + 阅读顺序
  ├─ 全页 OCR  PP-OCRv6 Small (OpenVINO)     → 文字行 + 坐标（一次推理，不逐区域裁剪）
  ├─ 表格结构  SLANet (ONNX Runtime)         → 行列结构（支持合并单元格 / 无线表格）
  ├─ 坐标分配  纯算法                         → OCR 行按中心点归入版面区域
  └─ Markdown  纯算法                         → 按阅读顺序拼接标题 / 正文 / 表格
```

**关键优化**：全页 OCR 只推理一次，再按坐标把文字分配到各版面区域，
避免「检测 N 个区域 → 裁剪 N 次 → OCR 推理 N 次」的重复开销。

---

## 📊 性能与测试结论

### 最终方案各阶段耗时（v3，单页 A4 150dpi，沙箱受限约 1 核，3 次平均）

| 阶段 | 模型 | 推理后端 | 耗时 | 占比 |
|------|------|---------|------|------|
| 版面检测 | PP-DocLayoutV3 (124.5MB) | OpenVINO | 722ms | 55% |
| 全页 OCR | PP-OCRv6 Small det+rec (30MB) | OpenVINO | 527ms | 40% |
| 表格结构识别 | SLANet (7.4MB) | ONNX Runtime | 74ms | 6% |
| 坐标分配 + Markdown 组装 | 纯算法 | — | <1ms | ~0% |
| **总计** | | | **1,323ms** | **0.76 pages/s** |

### 优化历程（同一测试文档）

| 版本 | 方案 | 单页耗时 | 加速比 |
|------|------|---------|--------|
| v0 | 裸 Paddle 逐区域推理 | 10,380ms | 1.0x（基线） |
| v1 | OpenVINO 版面 + ONNX 全页 OCR | 2,890ms | 3.6x |
| v2 | OpenVINO 全加速（无表格模型） | 1,139ms | 9.1x |
| **v3** | **OpenVINO + SLANet 表格（最终）** | **1,323ms** | **7.8x** |

> v2 → v3 多花约 184ms 引入 SLANet，换来完整的表格结构识别能力
> （合并单元格、无线表格），是精度与速度的平衡点。

### 关键结论

1. **OpenVINO 是 CPU 方案的核心加速器**：同模型下比裸 Paddle 推理快约 4–12 倍。
2. **全页 OCR 一次推理**是最大架构优化：从逐区域 N 次推理降到 1 次。
3. **检测分辨率 960px** 是 OCR 速度与完整度的最佳平衡点。
4. **文本抽取分流**：电子版 PDF（有文本层）走 PyMuPDF，实测约 **1090 pages/s**，
   只有扫描件才走 OCR。
5. **生产环境预估**（真实 4 核 Intel Xeon）：单页约 400–600ms，
   3 Worker 并发约 5–6 pages/s，日处理约 15–17 万页。

> 完整测试数据、踩坑记录见 [`phase.md`](phase.md)（Phase 1–7）。

---

## 📁 仓库结构

| 文件 | 说明 |
|------|------|
| `doc_parser_openvino.py` | **最终方案 v3** — OpenVINO + SLANet 全链路 pipeline |
| `doc_parser.py` | v0 裸 Paddle 版（对照基线） |
| `doc_parse_benchmark.py` | Tesseract + PyMuPDF 基准测试 |
| `paddleocr_benchmark.py` | PaddleOCR PP-OCRv6 裸推理测试 |
| `rapidocr_openvino.yaml` | RapidOCR OpenVINO 后端配置 |
| `download_models.sh` | 下载并转换版面检测模型 |
| `phase.md` | 完整测试记录（Phase 1–7） |
| `文档解析服务_技术报告.md` | 技术报告 + 多架构加速框架选型 |
| `models/SLANet_fixed.onnx` | 表格结构模型（已修复 Loop 算子） |
| `test_docs/` | 测试文档（含表格的合同页、扫描件等） |

---

## 🖥️ 多架构支持现状

当前默认链路（OpenVINO）在 **Intel CPU** 上性能最优。OpenVINO 同时也支持
ARM64 与 AMD（走通用 x86 路径），但在非 Intel 架构上无法发挥专属指令集优化，
性能会打折扣。各架构推荐后端：

| 架构 | 推荐加速框架 | 说明 |
|------|-------------|------|
| Intel Xeon / Core | **OpenVINO**（当前默认） | AVX2 / AVX-512 / AMX 深度优化，性能最强 |
| AMD EPYC / Ryzen | ONNX Runtime（+ ZenDNN） | 通用 x86 路径，可叠加 AMD ZenDNN |
| ARM64（鲲鹏 / 飞腾 / Ampere） | ONNX Runtime + ACL | ARM Compute Library 原生优化 |
| Apple Silicon | ONNX Runtime + CoreML | 可利用 ANE 神经引擎 |
| NVIDIA GPU | ONNX Runtime + CUDA / TensorRT | 比 CPU 快 5–20 倍（可选） |
| 海光 / 兆芯（x86 兼容） | ONNX Runtime | 本质 x86，缺 AVX-512 |

> 详细选型与部署准备清单见 [`文档解析服务_技术报告.md`](文档解析服务_技术报告.md)。

---

## 🗺️ 下一步计划（Roadmap）

**目标：在各类常见服务端架构上都支持高效运行。**

- [ ] **硬件自动检测 + 后端自动选择**：服务启动时探测 CPU 型号 / 指令集 / GPU，
      自动选用最优推理后端（OpenVINO / ONNX Runtime / CUDA）。
- [ ] **AMD 服务器适配与实测**：ONNX Runtime + ZenDNN，验证 EPYC / Ryzen 性能。
- [ ] **ARM64 服务器适配与实测**：鲲鹏 / 飞腾 / Ampere Altra，ONNX Runtime + ACL。
- [ ] **GPU 加速路径**：NVIDIA CUDA / TensorRT 后端，覆盖有显卡的客户。
- [ ] **统一 ONNX 模型格式**：一套模型文件适配所有后端，无需按架构单独转换。
- [ ] **Docker 多架构镜像**：`linux/amd64` + `linux/arm64` 一键部署。
- [ ] **部署验证脚本 `verify.py`**：环境检测、模型校验、单页推理 + 并发压测。
- [ ] **服务化封装**：FastAPI HTTP 接口，支持批量与并发 Worker。
- [ ] **精度与稳定性**：中文文档识别精度、复杂版面（多栏 / 跨页表格）、
      长时间运行内存泄漏检测。
- [ ] **INT8 量化**：OpenVINO NNCF 量化，进一步压缩模型、提升吞吐。

---

## 🐞 已知问题与修复

- **PaddlePaddle 3.3.1 OneDNN/PIR bug**：`ConvertPirAttribute2RuntimeAttribute not support`。
  改用 OpenVINO / ONNX Runtime 后端绕开，不依赖 Paddle 原生推理。
- **SLANet ONNX Loop 算子无法加载**：原始导出的 Loop 条件输入 shape 声明错误
  （推断 `[1]` vs 声明 `[]`）。修复方法：在 And 节点输出后插入 Squeeze 节点，
  生成 `models/SLANet_fixed.onnx`，并用 ONNX Runtime（禁用图优化）加载。

---

## 📄 许可

模型权重遵循 PaddlePaddle / PaddleX 官方许可（Apache-2.0）。
本仓库代码可自由使用、修改、商用。
