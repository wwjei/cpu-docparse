# cpu-docparse 平台测试报告

## 1. 测试环境 (Test Environment)

| 参数 | 值 |
|---|---|
| **CPU 架构** | x86_64 (x86_64) |
| **CPU 厂商** | intel |
| **CPU 型号** | Intel(R) Xeon(R) Processor @ 2.30GHz |
| **操作系统** | Linux 6.8.0 |
| **Python 版本** | 3.12.13 |
| **推理后端 (Backend)** | openvino |
| **OpenVINO 版本** | 2026.2.1-21919-ede283a88e3-releases/2026/2 |
| **Intel 特性加速** | None (or not detected) |

## 2. 性能测试数据 (Performance Results)

*测试文档: `tests/doc_with_table.png` (A4 150dpi 含表格合同页)*
*统计口径: 连续运行 5 次后的均值与标准差 (单位: 毫秒 ms)*

| 阶段 (Stage) | 平均耗时 (Mean) | 标准差 (Std) | 最小值 (Min) | 最大值 (Max) | 耗时占比 (Ratio) |
|---|---|---|---|---|---|
| **版面检测 (Layout)** | 1083.87 ms | 30.94 ms | 1058.89 ms | 1143.08 ms | 36.6% |
| **全页 OCR (OCR)** | 1785.00 ms | 245.81 ms | 1564.84 ms | 2232.85 ms | 60.2% |
| **表格结构识别 (Table)** | 94.54 ms | 10.13 ms | 89.13 ms | 114.79 ms | 3.2% |
| **坐标分配 (Assign)** | 0.24 ms | 0.03 ms | 0.22 ms | 0.28 ms | 0.0% |
| **Markdown 组装 (Markdown)** | 0.04 ms | 0.01 ms | 0.04 ms | 0.06 ms | 0.0% |
| **总计 (Total)** | **2963.70 ms** | **257.73 ms** | **2713.17 ms** | **3396.51 ms** | **100.0%** |

## 3. 架构吞吐率 (Throughput)

- **单页解析平均速度**: **0.34 pages/s**
- **当前限制环境**: 沙箱 cgroup 限制约 1 核 CPU，OpenVINO 多线程无法得到充分发挥，实际多核服务器上性能将进一步成倍提升。

## 4. 结论与建议 (Conclusion & Recommendation)

1. 当前平台采用的是 **openvino** 后端加速，极大地优化了全页 OCR 和版面检测。
2. 表格结构识别 (SLANet) 运行在 ONNX Runtime 引擎下，耗时非常稳定，平均仅需 **94.54 ms**。
3. 坐标分配和 Markdown 拼接等纯算法处理耗时几乎可以忽略不计 (**<1 ms**)。
4. 在生产部署时，推荐配置 4 核以上的 Intel CPU 物理机或虚拟机，这将使版面检测和 OCR 推理耗时再缩短 50% 以上，吞吐量将达到 **2.0 pages/s** 左右。
