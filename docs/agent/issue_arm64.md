## 目标

在 ARM64 CPU（鲲鹏 920 / 飞腾 S2500 / Ampere Altra / Apple M 系列）上验证 cpu-docparse 全链路性能。

## 背景

当前项目已在 Intel x86_64 上完成验证（~1,323ms）。ARM64 服务器在国内信创环境（鲲鹏、飞腾）中广泛使用，需要确认 ONNX Runtime 在 ARM64 上的兼容性和性能表现。

## 环境要求

- CPU: ARM64 (AArch64) — 鲲鹏 920 / 飞腾 S2500 / Ampere Altra / Apple M1+
- OS: Ubuntu 22.04 ARM64 / openEuler / Kylin
- Python: 3.10+（注意：部分环境需要源码编译）
- 内存: 8GB+

## 验证步骤

### 1. 克隆并安装

```bash
git clone https://github.com/wwjei/cpu-docparse.git
cd cpu-docparse
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**注意**：ARM64 上 `onnxruntime` 和 `opencv-python` 可能需要从源码编译或使用特定 wheel：
```bash
# 如果 pip install 失败，尝试：
pip install onnxruntime==1.17.0  # 检查是否有 aarch64 wheel
pip install opencv-python-headless  # headless 版本兼容性更好
```

### 2. 下载模型

```bash
bash scripts/download_models.sh
```

如果 PaddleX 在 ARM64 上不可用，可直接从 Release 或网盘下载预转换的 ONNX 模型文件放入 `models/` 目录。

### 3. 修改推理后端

同 AMD 验证，需要：
- Layout 检测：OpenVINO → ONNX Runtime
- OCR：RapidOCR OpenVINO 后端 → ONNX Runtime 后端
- SLANet：已是 ONNX Runtime，无需修改

### 4. 可选：ARM 加速

- **ARM Compute Library (ACL)**：ONNX Runtime 支持 ACL EP 加速
  ```bash
  pip install onnxruntime  # 确认是否包含 ACL 支持
  # 或从源码编译 onnxruntime with --use_acl
  ```
- **XNNPACK**：ONNX Runtime 内置，ARM 上默认可能启用

### 5. 运行测试

```bash
python -m cpu_docparse tests/doc_with_table.png --json
```

### 6. 记录数据

| 阶段 | 耗时 (ms) | 备注 |
|------|-----------|------|
| Layout 检测 (PP-DocLayoutV3) | ? | ONNX Runtime |
| OCR 文字检测 (PP-OCRv6 det) | ? | |
| OCR 文字识别 (PP-OCRv6 rec) | ? | |
| 表格结构 (SLANet) | ? | ONNX Runtime |
| Markdown 构建 | ? | |
| **总计** | ? | |

记录 CPU 型号、核心数、是否启用 ACL/XNNPACK。

## 输出要求

1. 更新 README.md 中 "ARM64" 行的性能数据
2. 在 `docs/phase.md` 中追加 ARM64 验证记录
3. 如有代码修改，提交 PR
4. 将验证结论回复到本 Issue

## 已知问题

- SLANet ONNX 需要 `ORT_DISABLE_ALL` 图优化级别
- 鲲鹏 920 为 ARMv8.2，部分新指令可能不支持
- Apple M 系列在 Linux (Asahi) 下 ONNX Runtime 支持可能有限
- RapidOCR 的 OpenVINO 后端在 ARM 上不可用，必须切换为 onnxruntime 后端
