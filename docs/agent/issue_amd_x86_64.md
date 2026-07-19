## 目标

在 AMD x86_64 CPU（EPYC / Ryzen）上验证 cpu-docparse 全链路性能，确认非 Intel CPU 下的可用性和耗时基线。

## 背景

当前项目已在 Intel x86_64 (Xeon) 上完成验证，使用 OpenVINO 作为推理后端，全链路耗时 ~1,323ms。AMD CPU 不支持 OpenVINO 的 Intel 专用优化（如 AMX/VNNI），需要切换到 ONNX Runtime（可选 ZenDNN 插件）作为推理后端。

## 环境要求

- CPU: AMD EPYC 7003/9004 系列 或 Ryzen 5000/7000 系列（至少 4 核）
- OS: Ubuntu 22.04+ / CentOS 8+
- Python: 3.10+
- 内存: 8GB+

## 验证步骤

### 1. 克隆并安装

```bash
git clone https://github.com/wwjei/cpu-docparse.git
cd cpu-docparse
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. 下载模型

```bash
bash scripts/download_models.sh
```

### 3. 修改推理后端

当前 `cpu_docparse/parser.py` 中：
- Layout 检测使用 OpenVINO（`openvino.runtime`）
- OCR 使用 RapidOCR + OpenVINO 后端

AMD 环境需要：
- 将 Layout 检测改为 ONNX Runtime 推理（模型已是 ONNX 格式）
- RapidOCR 配置改为 `rapidocr_onnxruntime` 后端（修改 `rapidocr_openvino.yaml` 或新建 `rapidocr_onnxruntime.yaml`）
- SLANet 已使用 ONNX Runtime，无需修改

具体修改点：
```python
# parser.py _detect_layout 方法中
# 替换 OpenVINO InferenceEngine 为 onnxruntime.InferenceSession
import onnxruntime as ort
session = ort.InferenceSession("models/PP-DocLayoutV3.onnx")
```

### 4. 运行测试

```bash
python -m cpu_docparse tests/doc_with_table.png --json
```

### 5. 记录数据

请记录以下指标并更新到 README.md 的性能表格中：

| 阶段 | 耗时 (ms) | 备注 |
|------|-----------|------|
| Layout 检测 (PP-DocLayoutV3) | ? | ONNX Runtime |
| OCR 文字检测 (PP-OCRv6 det) | ? | |
| OCR 文字识别 (PP-OCRv6 rec) | ? | |
| 表格结构 (SLANet) | ? | ONNX Runtime |
| Markdown 构建 | ? | |
| **总计** | ? | |

### 6. 可选：ZenDNN 加速

如果环境支持，尝试安装 AMD ZenDNN 插件：
```bash
pip install onnxruntime-zendnn
```
对比启用/未启用 ZenDNN 的性能差异。

## 输出要求

1. 更新 README.md 中 "AMD x86_64" 行的性能数据
2. 在 `docs/phase.md` 中追加 AMD 验证记录
3. 如有代码修改（后端切换），提交 PR 并说明
4. 将验证结论回复到本 Issue

## 已知问题

- SLANet ONNX 模型需要 `graph_optimization_level = ORT_DISABLE_ALL` 才能加载（Loop 算子 bug）
- PP-DocLayoutV3 输入尺寸 800x800，需 resize + padding
