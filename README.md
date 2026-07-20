# cpu-docparse

**极简 CPU 文档解析库** — 将扫描件/图片解析为结构化 Markdown，全 CPU 推理，无需 GPU。

当前范围：**OCR 扫描件/图片链路**（版面检测 → 阅读顺序 → 表格结构 → 文字识别 → Markdown）。

---

## 快速开始

```bash
# 1. 安装
git clone https://github.com/wwjei/cpu-docparse.git
cd cpu-docparse
pip install -e .              # 通用安装 (ONNX Runtime, 适用 AMD/ARM/Intel)
pip install -e ".[intel]"     # Intel CPU 推荐 (额外启用 OpenVINO 加速)

# 2. 获取版面检测模型 (约125MB, 超GitHub限制未入库)
pip install paddlepaddle paddle2onnx paddlex   # 仅转换时需要
bash scripts/download_models.sh

# 3. 使用
python -m cpu_docparse tests/doc_with_table.png
```

> **推理后端自动选择**：库会根据 CPU 自动挑选最优引擎 —— Intel CPU 用 OpenVINO
> (AMX/VNNI 加速)，AMD/ARM/其他用 ONNX Runtime。无需手动配置。
> 也可用 `DocParser(backend="onnxruntime")` 或 CLI `--backend` 强制指定。

### 作为库调用

```python
from cpu_docparse import DocParser

parser = DocParser()
result = parser.parse("scan.png")

print(result["markdown"])    # 结构化 Markdown
print(result["regions"])     # 版面区域 (label/bbox/score/order/content)
print(result["timings"])     # 各阶段耗时
```

### CLI

```bash
python -m cpu_docparse scan.png              # 输出 Markdown 到 stdout
python -m cpu_docparse scan.png -o out.md    # 写入文件
python -m cpu_docparse scan.png --json       # 完整 JSON 输出
python -m cpu_docparse scan.png -q           # 静默模式
```

---

## 项目结构

```
cpu-docparse/
├── cpu_docparse/           # 库代码 (pip install -e . 后可 import)
│   ├── __init__.py         # 入口: from cpu_docparse import DocParser
│   ├── __main__.py         # CLI: python -m cpu_docparse
│   ├── backend.py          # 推理后端自动选择 (Intel→OpenVINO, 其他→ONNX Runtime)
│   └── parser.py           # 核心解析器
├── models/                 # 模型文件
│   └── SLANet_fixed.onnx   # 表格结构模型 (7.5MB, 已入库)
├── benchmarks/             # 性能测试脚本 (历史对照)
│   ├── doc_parse_benchmark.py
│   ├── doc_parser_v0_paddle.py
│   └── paddleocr_benchmark.py
├── docs/                   # 文档
│   ├── phase.md            # 完整测试记录 (Phase 1-7)
│   ├── 文档解析服务_技术报告.md
│   └── agent/              # agent 协作指南 + 架构验证 Issue 模板
├── scripts/
│   └── download_models.sh  # 下载 PP-DocLayoutV3 并转 ONNX
├── tests/                  # 测试文档
├── rapidocr_openvino.yaml  # OCR 后端配置 (Intel/OpenVINO)
├── rapidocr_onnxruntime.yaml  # OCR 后端配置 (AMD/ARM/ONNX Runtime)
├── pyproject.toml          # 包定义
└── README.md
```

---

## 解析链路

```
输入图片 (扫描件/截图)
  ├─ 版面检测  PP-DocLayoutV3                → 25 类区域 + bbox + 阅读顺序
  ├─ 全页 OCR  PP-OCRv6 Small                → 文字行 + 坐标 (一次推理)
  ├─ 表格结构  SLANet (ONNX Runtime)         → 行列结构 (支持合并单元格)
  ├─ 坐标分配  纯算法                         → OCR 行归入版面区域
  └─ Markdown  纯算法                         → 按阅读顺序拼接输出

推理后端 (自动选择):
  Intel CPU → OpenVINO (AMX/VNNI)   |   AMD/ARM/其他 → ONNX Runtime
```

---

## 已验证架构性能

测试文档: A4 150dpi 含表格合同页 (`tests/doc_with_table.png`)

### x86_64 Intel (已验证 ✅)

| 阶段 | 模型 | 后端 | 耗时 | 占比 |
|------|------|------|------|------|
| 版面检测 | PP-DocLayoutV3 (124.5MB) | OpenVINO | 722ms | 55% |
| 全页 OCR | PP-OCRv6 Small (30MB) | OpenVINO | 527ms | 40% |
| 表格结构 | SLANet (7.4MB) | ONNX Runtime | 74ms | 6% |
| 坐标分配 + Markdown | 纯算法 | — | <1ms | ~0% |
| **总计** | | | **1,323ms** | **0.76 pages/s** |

> 测试环境: 沙箱受限约 1 核 Intel Xeon, 8GB RAM
> 生产预估 (4核): 单页 400-600ms, 3 Worker 并发 5-6 pages/s

### AMD x86_64 (已验证 ✅)

| 阶段 | 模型 | 后端 | 耗时 | 占比 |
|------|------|------|------|------|
| 版面检测 | PP-DocLayoutV3 (124.5MB) | ONNX Runtime | 508ms | 25% |
| 全页 OCR | PP-OCRv6 Small (30MB) | ONNX Runtime | 1,423ms | 70% |
| 表格结构 | SLANet (7.4MB) | ONNX Runtime | 91ms | 5% |
| 坐标分配 + Markdown | 纯算法 | — | <1ms | ~0% |
| **总计** | | | **2,022ms** | **0.49 pages/s** |

> 测试环境: Ubuntu 22.04, AMD EPYC 9334 32-Core (容器限 1 核), 63GB RAM, ONNX Runtime 1.23.2
> 5 次平均 (warmup 2 次后): min=1,858ms / max=2,202ms / avg=2,022ms
> 复现: `python benchmarks/run_benchmark.py tests/doc_with_table.png --runs 5 --warmup 2`
> 详见 [Issue #1](https://github.com/wwjei/cpu-docparse/issues/1)

### ARM64 / aarch64 (待验证 ⏳)

→ 见 [Issue #2](https://github.com/wwjei/cpu-docparse/issues/2)

---

## 多架构支持计划

**同一份代码，自动适配。** 推理后端由 `cpu_docparse/backend.py` 根据 CPU 自动选择，
无需为不同架构维护分支代码。Intel 走 OpenVINO，其余走 ONNX Runtime。

| 架构 | 自动选择后端 | 加速特性 | 状态 | Issue |
|------|---------|---------|------|-------|
| Intel x86_64 | OpenVINO | AMX / VNNI | ✅ 已验证 | — |
| AMD x86_64 (EPYC/Ryzen) | ONNX Runtime | ZenDNN (可选) | ✅ 已验证 | #1 |
| ARM64 (鲲鹏/飞腾/Ampere/Apple M) | ONNX Runtime | ACL / XNNPACK | ⏳ 待验证 | #2 |
| NVIDIA GPU (可选加速) | ONNX Runtime | CUDA EP | 📋 规划中 | #3 |

**各架构运行配置**：

| 架构 | 版面检测 | OCR | 表格结构 | 实测单页耗时 |
|------|---------|-----|---------|-------------|
| Intel x86_64 | OpenVINO | OpenVINO | ONNX Runtime | ~1,323ms |
| AMD x86_64 | ONNX Runtime | ONNX Runtime | ONNX Runtime | ~2,022ms |
| ARM64 (预估) | ONNX Runtime | ONNX Runtime | ONNX Runtime | ~1,800-2,500ms |

> ARM64 预估基于 ONNX Runtime 通用 CPU 路径，实际数据以
> [Issue #2](https://github.com/wwjei/cpu-docparse/issues/2) 验证结果为准。

**目标: 在各类常见服务端架构上都支持高效运行。**

---

## 依赖

运行期:
```
openvino>=2024.0
onnxruntime>=1.17
rapidocr[openvino]>=1.4
opencv-python>=4.8
numpy>=1.24
pillow>=10.0
```

模型下载/转换 (仅首次):
```
paddlepaddle>=3.0
paddle2onnx>=1.2
paddlex>=3.0
```

---

## 已知问题

- **PP-DocLayoutV3.onnx (125MB)** 超 GitHub 100MB 限制，需运行 `scripts/download_models.sh` 获取
- **SLANet ONNX Loop 算子**: 原始导出有 shape 声明 bug，已修复为 `SLANet_fixed.onnx`
- **PaddlePaddle 3.3.1 OneDNN/PIR bug**: 本项目不依赖 Paddle 原生推理，使用 OpenVINO/ONNX Runtime 绕开

---

## 许可

Apache-2.0。模型权重遵循 PaddlePaddle/PaddleX 官方许可。
