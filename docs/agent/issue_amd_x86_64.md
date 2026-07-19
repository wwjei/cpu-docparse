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
pip install -e .   # 通用安装即可，AMD 自动走 ONNX Runtime
```

### 2. 下载模型

```bash
bash scripts/download_models.sh
```

### 3. 确认后端自动选择

代码已支持自动选择后端（`cpu_docparse/backend.py`），AMD CPU 会自动走 ONNX Runtime，
**无需修改代码**。验证一下：

```bash
python -c "from cpu_docparse.backend import select_backend, get_backend_info; \
  b=select_backend(); print(b.value); print(get_backend_info(b))"
# 期望输出: onnxruntime
```

如需强制指定后端对比：`DocParser(backend="onnxruntime")` 或 CLI `--backend onnxruntime`。

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

1. 更新 README.md 中 "AMD x86_64" 行的性能数据（替换预估值）
2. 在 `docs/phase.md` 中追加 AMD 验证记录
3. 如发现后端自动选择有误或需调优，提交 PR 并说明
4. 将验证结论回复到本 Issue

## 已知问题

- SLANet ONNX 模型需要 `graph_optimization_level = ORT_DISABLE_ALL` 才能加载（Loop 算子 bug）
- PP-DocLayoutV3 输入尺寸 800x800，需 resize + padding
