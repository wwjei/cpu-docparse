## 目标

验证 NVIDIA GPU 加速方案（ONNX Runtime + CUDA EP），为有 GPU 环境的客户提供可选的高性能部署路径。

## 背景

当前项目定位为 CPU 极简方案，但部分客户环境有 NVIDIA GPU（T4/A10/3090 等）。验证 GPU 加速可作为可选增强，不改变 CPU-only 的默认路径。

## 环境要求

- GPU: NVIDIA T4 / A10 / RTX 3090 / RTX 4090（4GB+ 显存）
- Driver: 525+
- CUDA: 11.8 / 12.x
- cuDNN: 8.6+
- OS: Ubuntu 22.04
- Python: 3.10+

## 验证步骤

### 1. 安装 GPU 版 ONNX Runtime

```bash
pip install onnxruntime-gpu  # 替代 onnxruntime
```

### 2. 修改推理代码

在 `parser.py` 中为 ONNX Runtime session 添加 CUDA EP：

```python
import onnxruntime as ort

# 优先 CUDA，回退 CPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession(model_path, providers=providers)
```

对三个模型分别测试：
- PP-DocLayoutV3.onnx（Layout 检测）
- SLANet_fixed.onnx（表格结构）
- PP-OCRv6 det/rec（通过 RapidOCR 或直接加载）

### 3. 注意事项

- SLANet 仍需 `ORT_DISABLE_ALL` 图优化级别
- 首次推理会有 CUDA kernel 编译开销（warm-up），需跑 3 次取后两次平均
- 小模型（SLANet 7.4MB）GPU 加速收益可能不明显，重点观察 Layout 和 OCR

### 4. 运行测试

```bash
# warm-up
python -m cpu_docparse tests/doc_with_table.png --json
python -m cpu_docparse tests/doc_with_table.png --json
# 正式记录
python -m cpu_docparse tests/doc_with_table.png --json
```

### 5. 记录数据

| 阶段 | CPU 耗时 (ms) | GPU 耗时 (ms) | 加速比 |
|------|--------------|--------------|--------|
| Layout 检测 | 参考 x86 数据 | ? | ? |
| OCR 检测 | | ? | ? |
| OCR 识别 | | ? | ? |
| 表格结构 (SLANet) | | ? | ? |
| **总计** | ~1,323 | ? | ? |

记录 GPU 型号、显存占用、CUDA 版本。

### 6. 多图片批量测试

```bash
for img in tests/scan_page_*.png; do
  python -m cpu_docparse "$img" --json -q
done
```

## 输出要求

1. 在 README.md 中补充 GPU 加速可选方案说明
2. 在 `docs/phase.md` 中追加 GPU 验证记录
3. 如代码修改合理（如添加 providers 配置参数），提交 PR
4. 将验证结论回复到本 Issue

## 设计原则

- GPU 为可选增强，不破坏 CPU-only 默认路径
- 通过配置/参数切换，不硬编码 CUDA
- 无 GPU 时自动回退 CPU，不报错
