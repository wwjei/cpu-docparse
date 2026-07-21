# AMD ZenDNN 加速可行性分析

> 记录时间: 2026-07-21
> 背景: AMD x86_64 (EPYC 9334) 验证完成后，评估 ZenDNN 是否能进一步优化 ONNX Runtime 推理性能。

## 结论速览

**ZenDNN 目前没有可用的 ONNX Runtime 集成路径，本项目暂不采用。**

实际优化空间在 ONNX Runtime 自身的线程调优和模型量化上。

## 详细分析

### 1. ZenDNN 是什么

ZenDNN (Zen Deep Neural Network) 是 AMD 推出的深度学习推理加速库，
针对 AMD EPYC CPU 优化，提供 Convolution / MatMul / Elementwise / Pool /
Gelu / LayerNorm 等优化算子。

### 2. 插件生态现状

| ZenDNN 版本 | 发布年份 | ONNX Runtime 插件 | PyTorch 插件 | TensorFlow 插件 |
|------------|---------|------------------|--------------|-----------------|
| 4.0 | 2022 | ✅ v1.12.1 | ✅ | ✅ |
| 5.0 | 2024 | ❌ 停止维护 | ✅ (zentorch) | ✅ (zentf) |
| 5.1 | 2025 | ❌ | ✅ | ✅ |
| 5.2 | 2025 | ❌ | ✅ | ✅ |

**关键事实**: 从 ZenDNN 5.0 起，AMD 把精力转向 PyTorch / TensorFlow 插件，
**ONNX Runtime 插件已停止维护**。最后一个支持 ORT 的版本是 4.0 (2022)，
绑定 ONNX Runtime v1.12.1。

### 3. 与本项目的兼容性

本项目当前环境:
- ONNX Runtime: **1.23.2**
- 模型格式: ONNX (PP-DocLayoutV3 / PP-OCRv6 / SLANet)

ZenDNN 4.0 要求 ORT v1.12.1，与本项目 1.23.2 差距过大，强装大概率不兼容。
**无法在不降级 ORT 的前提下使用 ZenDNN 加速现有 ONNX 模型。**

### 4. CPU 兼容性

ZenDNN 官方仅支持 **AMD EPYC 服务器 CPU** (Zen 3/4/5)，
不背书 Ryzen 消费级 (虽然架构相同)。

测试机 EPYC 9334 (Zen 4) 在支持列表内，但插件生态问题使其无法落地。

## 替代优化方向 (未实施，作为后续 Issue 参考)

### 方向一: ONNX Runtime 线程调优 (零成本，最快验证)

```bash
# 试不同线程数对比
OMP_NUM_THREADS=8  python benchmarks/run_benchmark.py tests/doc_with_table.png --runs 3 --warmup 1
OMP_NUM_THREADS=16 python benchmarks/run_benchmark.py tests/doc_with_table.png --runs 3 --warmup 1
```

或在 SessionOptions 中显式配置:
```python
opts.intra_op_num_threads = 4   # 匹配可用核心
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
```

当前 RapidOCR 配置 `intra_op_num_threads: 4`，容器有 16 线程可见，有调优空间。

### 方向二: 模型量化 (INT8，精度换速度)

OCR 模型量化后通常能快 30-50%，但需验证识别精度:

```bash
pip install onnxruntime-tools
python -m onnxruntime.quantization.preprocess \
  --input models/PP-DocLayoutV3.onnx \
  --output models/PP-DocLayoutV3_pre.onnx
# 然后做 dynamic quantization
```

### 方向三: 走 PyTorch 路径用 zentorch (改造量大，不建议)

如果一定要用 ZenDNN，只能把 PaddleOCR 模型用 PyTorch 版本重跑，
再用 `zentorch` 加速。这不是"基于已有 ONNX"，改造量大，
现阶段不建议。

## 当前 AMD 性能基线 (未优化)

| 阶段 | 耗时 | 占比 |
|------|------|------|
| 版面检测 (PP-DocLayoutV3) | 508ms | 25% |
| 全页 OCR (PP-OCRv6) | 1,423ms | 70% |
| 表格结构 (SLANet) | 91ms | 5% |
| **总计** | **2,022ms** | 0.49 pages/s |

> 测试环境: AMD EPYC 9334 32-Core (容器限 1 核), Ubuntu 22.04, ONNX Runtime 1.23.2
> OCR 是主要瓶颈 (70%)，优化应优先针对 OCR 阶段。

## Intel vs AMD OCR 性能差异分析

### 数据对比

| | OCR 耗时 | 后端 | CPU | 备注 |
|---|---------|------|-----|------|
| Intel | 731ms | OpenVINO | Xeon (AMX/VNNI) | 沙箱受限 ~1 核, 8GB |
| AMD | 1,423ms | ONNX Runtime | EPYC 9334 (Zen 4) | 容器限 1 核, 63GB |

OCR 阶段差距约 2 倍，但 layout 检测 AMD 反而更快 (508ms vs 799ms)。

### 原因拆解

**1. OpenVINO 在 Intel 上有专用优化路径（主因）**

OpenVINO 是 Intel 自家推理引擎，对 Intel CPU 做了深度优化：
- 算子融合: Conv+BN+ReLU 等常见模式融合为单个 kernel，减少内存读写
- 图优化更激进: 常量折叠、layout 转换 (NCHW↔NHWC) 选最优内存排布
- LATENCY 性能模式: 配置 `performance_hint: LATENCY`，针对低延迟专门调度

ONNX Runtime 是通用引擎，跨所有硬件，优化深度天然不如"亲儿子"。

**2. AMX / VNNI 指令集加速（关键加成）**

Intel Xeon 具备 `amx_int8`、`avx512_vnni` 指令。OCR 模型大量是矩阵乘和卷积，
VNNI/AMX 能让 INT8/低精度运算吞吐量翻倍甚至更多。

AMD EPYC 9334 (Zen 4) 有 AVX-512 但没有 AMX，VNNI 支持有限。
这是架构层面的硬差距。

**3. 对比本身"不公平"**

Intel 数据在受限沙箱 (~1 核, 8GB) 跑出，AMD 是真实 EPYC 硬件。
按理 AMD 硬件更强，OCR 却慢一倍——恰恰说明 OpenVINO + Intel 指令集的
组合优势压过了硬件差距。

**4. 为什么 layout 检测 AMD 反而更快**

layout 模型在两边都走相对"干净"的单次推理路径，算子融合红利小，
此时 AMD EPYC 单核频率/缓存优势体现出来 (508ms vs 799ms)。
而 OCR 是检测+识别两阶段、大量小算子，OpenVINO 的算子融合和调度优化红利最大。

### 一句话总结

OCR 慢不是因为 AMD 硬件差，而是 OpenVINO 在 Intel 上对 OCR 这类负载有
专用算子融合 + AMX/VNNI 指令集双重加成，ONNX Runtime 在 AMD 上只能走通用路径，
吃不到这些红利。这也解释了为什么 ZenDNN 停掉 ORT 插件是个遗憾——
如果 AMD 有对等的 ORT 优化层，这个差距能缩小很多。

---

## 参考资料

- ZenDNN 官方: https://www.amd.com/zh-cn/developer/zendnn.html
- ZenDNN GitHub: https://github.com/amd/ZenDNN
- ZenDNN 历史版本: https://www.amd.com/en/developer/zendnn/zendnn-archives.html
- ZenDNN 4.2 插件架构: https://www.amd.com/zh-cn/developer/resources/technical-articles/supercharge-your-ai-inference-with-zendnn-on-amd-epyc-cpus.html
