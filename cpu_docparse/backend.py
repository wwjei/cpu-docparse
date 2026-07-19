"""
推理后端自动选择模块

根据 CPU 架构和可用库自动选择最优推理引擎：
- Intel x86_64 (有 AMX/VNNI) → OpenVINO
- AMD x86_64 → ONNX Runtime (可选 ZenDNN)
- ARM64 → ONNX Runtime (可选 ACL/XNNPACK)
- 其他 → ONNX Runtime
"""

import platform
import subprocess
from enum import Enum
from typing import Optional


class Backend(Enum):
    OPENVINO = "openvino"
    ONNXRUNTIME = "onnxruntime"


def detect_cpu_vendor() -> str:
    """检测 CPU 厂商: intel / amd / arm / unknown"""
    machine = platform.machine().lower()
    if machine in ("aarch64", "arm64"):
        return "arm"

    try:
        out = subprocess.check_output(
            ["lscpu"], stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        if "GenuineIntel" in out:
            return "intel"
        if "AuthenticAMD" in out:
            return "amd"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # fallback: 读 /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            content = f.read()
        if "GenuineIntel" in content:
            return "intel"
        if "AuthenticAMD" in content:
            return "amd"
    except OSError:
        pass

    return "unknown"


def detect_intel_features() -> list:
    """检测 Intel CPU 加速特性 (AMX, VNNI 等)"""
    features = []
    try:
        out = subprocess.check_output(
            ["lscpu"], stderr=subprocess.DEVNULL, text=True, timeout=5
        )
        for line in out.splitlines():
            if line.startswith("Flags:"):
                flags = line.split(":", 1)[1].split()
                for feat in ("amx_tile", "amx_int8", "amx_bf16", "avx512_vnni", "avx_vnni"):
                    if feat in flags:
                        features.append(feat)
                break
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return features


def _openvino_available() -> bool:
    try:
        import openvino  # noqa: F401
        return True
    except ImportError:
        return False


def _onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def select_backend(preference: Optional[str] = None) -> Backend:
    """选择最优推理后端

    Args:
        preference: 用户指定 "openvino" / "onnxruntime" / None(auto)

    Returns:
        Backend enum
    """
    if preference == "openvino":
        if not _openvino_available():
            raise RuntimeError("指定了 openvino 后端但未安装 openvino 包")
        return Backend.OPENVINO

    if preference == "onnxruntime":
        if not _onnxruntime_available():
            raise RuntimeError("指定了 onnxruntime 后端但未安装 onnxruntime 包")
        return Backend.ONNXRUNTIME

    # auto 模式: 根据环境选择
    vendor = detect_cpu_vendor()

    if vendor == "intel" and _openvino_available():
        return Backend.OPENVINO

    if _onnxruntime_available():
        return Backend.ONNXRUNTIME

    if _openvino_available():
        return Backend.OPENVINO

    raise RuntimeError(
        "未找到可用的推理后端。请安装 openvino 或 onnxruntime:\n"
        "  pip install openvino    # Intel CPU 推荐\n"
        "  pip install onnxruntime # AMD/ARM/通用"
    )


def get_backend_info(backend: Backend) -> dict:
    """获取后端环境信息，用于日志和记录"""
    info = {
        "backend": backend.value,
        "cpu_vendor": detect_cpu_vendor(),
        "arch": platform.machine(),
    }

    if backend == Backend.OPENVINO:
        try:
            import openvino
            info["openvino_version"] = openvino.__version__
        except ImportError:
            pass
        info["intel_features"] = detect_intel_features()

    if backend == Backend.ONNXRUNTIME:
        try:
            import onnxruntime as ort
            info["onnxruntime_version"] = ort.__version__
            info["available_providers"] = ort.get_available_providers()
        except ImportError:
            pass

    return info
