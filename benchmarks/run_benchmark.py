#!/usr/bin/env python3
"""
多架构性能基准测试

用法:
    python benchmarks/run_benchmark.py [图片路径] [--runs 5] [--warmup 2]

默认测试 tests/doc_with_table.png, warmup 2 次后正式跑 5 次取平均。
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def collect_system_info() -> dict:
    """收集机器配置: CPU 型号/核心数/频率、内存、OS 等"""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "arch": platform.machine(),
        "cpu_model": "unknown",
        "cpu_sockets": None,
        "cpu_cores_physical": None,
        "cpu_threads": os.cpu_count(),
        "cpu_max_mhz": None,
        "mem_total_gb": None,
    }

    # lscpu (Linux)
    try:
        out = subprocess.check_output(["lscpu"], text=True, timeout=5)
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "Model name":
                info["cpu_model"] = v
            elif k == "Socket(s)":
                info["cpu_sockets"] = int(v)
            elif k == "Core(s) per socket":
                info["cpu_cores_per_socket"] = int(v)
            elif k == "CPU(s)":
                info["cpu_threads"] = int(v)
            elif k == "CPU max MHz":
                info["cpu_max_mhz"] = round(float(v))
        # 物理核 = sockets × cores_per_socket (虚拟化环境可能不准, 取 CPU(s) 兜底)
        sockets = info.get("cpu_sockets") or 1
        cores_per = info.get("cpu_cores_per_socket") or 1
        calc = sockets * cores_per
        info["cpu_cores_physical"] = max(calc, info["cpu_threads"] or 1) if calc > 1 else (info["cpu_threads"] or 1)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # /proc/cpuinfo fallback for model name
    if info["cpu_model"] == "unknown":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

    # 内存
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["mem_total_gb"] = round(kb / 1024 / 1024, 1)
                    break
    except (OSError, ValueError):
        pass

    return info


def main():
    parser = argparse.ArgumentParser(description="cpu-docparse 性能基准测试")
    parser.add_argument("image", nargs="?", default="tests/doc_with_table.png", help="测试图片")
    parser.add_argument("--runs", type=int, default=5, help="正式运行次数 (默认 5)")
    parser.add_argument("--warmup", type=int, default=2, help="预热次数 (默认 2)")
    parser.add_argument("--backend", default="auto", choices=["auto", "openvino", "onnxruntime"])
    args = parser.parse_args()

    from cpu_docparse import DocParser
    from cpu_docparse.backend import get_backend_info, select_backend

    backend = select_backend(args.backend if args.backend != "auto" else None)
    info = get_backend_info(backend)
    print(f"后端: {info['backend']}  CPU: {info['cpu_vendor']}/{info['arch']}")
    if "intel_features" in info and info["intel_features"]:
        print(f"加速特性: {', '.join(info['intel_features'])}")
    if "onnxruntime_version" in info:
        print(f"ONNX Runtime: {info['onnxruntime_version']}")
    if "openvino_version" in info:
        print(f"OpenVINO: {info['openvino_version']}")
    print(f"测试图片: {args.image}")
    print(f"预热 {args.warmup} 次, 正式 {args.runs} 次\n")

    # 初始化 (不计入推理时间)
    t0 = time.perf_counter()
    dp = DocParser(backend=args.backend, verbose=False)
    init_time = time.perf_counter() - t0
    print(f"初始化耗时: {init_time*1000:.0f}ms\n")

    # Warmup
    for i in range(args.warmup):
        dp.parse(args.image)
        print(f"  warmup {i+1}/{args.warmup} done")

    # 正式运行
    print()
    all_timings = []
    all_totals = []

    for i in range(args.runs):
        result = dp.parse(args.image)
        t = result["timings"]
        total = result["total_time"]
        all_timings.append(t)
        all_totals.append(total)
        print(f"  run {i+1}: layout={t['layout']*1000:.0f}ms  ocr={t['ocr']*1000:.0f}ms  "
              f"table={t['table']*1000:.0f}ms  total={total*1000:.0f}ms")

    # 统计
    n = len(all_totals)
    avg_total = sum(all_totals) / n
    min_total = min(all_totals)
    max_total = max(all_totals)

    stages = ["layout", "ocr", "assign", "table", "markdown"]
    avg_stages = {}
    for s in stages:
        vals = [t[s] for t in all_timings]
        avg_stages[s] = sum(vals) / n

    print(f"\n{'='*60}")
    print(f"结果 ({n} 次平均)")
    print(f"{'='*60}")
    print(f"{'阶段':<12} {'平均(ms)':>10} {'占比':>8}")
    print(f"{'-'*32}")
    for s in stages:
        ms = avg_stages[s] * 1000
        pct = avg_stages[s] / avg_total * 100 if avg_total > 0 else 0
        print(f"{s:<12} {ms:>10.1f} {pct:>7.1f}%")
    print(f"{'-'*32}")
    print(f"{'总计':<12} {avg_total*1000:>10.1f}")
    print(f"\nmin={min_total*1000:.0f}ms  max={max_total*1000:.0f}ms  avg={avg_total*1000:.0f}ms")
    print(f"吞吐: {1/avg_total:.2f} pages/s")

    # 机器配置
    sysinfo = collect_system_info()
    print(f"\n{'='*60}")
    print("机器配置")
    print(f"{'='*60}")
    print(f"OS:        {sysinfo['os']}")
    print(f"Python:    {sysinfo['python']}")
    print(f"CPU:       {sysinfo['cpu_model']}")
    cores_desc = []
    if sysinfo["cpu_cores_physical"]:
        cores_desc.append(f"{sysinfo['cpu_cores_physical']} 物理核")
    if sysinfo["cpu_threads"]:
        cores_desc.append(f"{sysinfo['cpu_threads']} 线程")
    if cores_desc:
        print(f"核心:      {' / '.join(cores_desc)}")
    if sysinfo["cpu_max_mhz"]:
        print(f"最大频率:  {sysinfo['cpu_max_mhz']} MHz")
    if sysinfo["mem_total_gb"]:
        print(f"内存:      {sysinfo['mem_total_gb']} GB")

    # 输出 JSON 方便记录
    summary = {
        "system": sysinfo,
        "backend": info,
        "image": args.image,
        "warmup": args.warmup,
        "runs": args.runs,
        "init_ms": round(init_time * 1000, 1),
        "avg_stages_ms": {k: round(v * 1000, 1) for k, v in avg_stages.items()},
        "avg_total_ms": round(avg_total * 1000, 1),
        "min_total_ms": round(min_total * 1000, 1),
        "max_total_ms": round(max_total * 1000, 1),
        "pages_per_sec": round(1 / avg_total, 2),
    }
    print(f"\n--- JSON ---")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
