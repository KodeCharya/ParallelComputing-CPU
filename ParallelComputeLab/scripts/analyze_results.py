#!/usr/bin/env python3
"""
analyze_results.py
Performs statistical analysis, speedup and efficiency calculation, table formatting,
and publication-quality plot generation from raw benchmark results.
"""

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT_LIBS = True
except ImportError:
    HAS_PLOT_LIBS = False


def parse_benchmark_name(full_name: str) -> dict:
    """
    Parses benchmark names such as:
    BM_VectorAdd_Parallel/262144/4/real_time
    BM_MatMul_Parallel/256/8/iterations:3
    BM_Memory_StridedRead/4194304/16
    """
    clean_name = re.sub(r'/(real_time|cpu_time|manual_time).*$', '', full_name)
    parts = clean_name.split('/')
    kernel_tag = parts[0]
    args = []
    for p in parts[1:]:
        if p.isdigit():
            args.append(int(p))
        else:
            try:
                args.append(float(p))
            except ValueError:
                args.append(p)
    return {
        "kernel_tag": kernel_tag,
        "args": args,
        "clean_name": clean_name
    }


def compute_statistics(values: list[float]) -> dict:
    if not values:
        return {"median": 0.0, "mean": 0.0, "std": 0.0, "iqr": 0.0, "min": 0.0, "max": 0.0}
    vals = sorted(values)
    n = len(vals)
    median = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    mean = sum(vals) / n
    variance = sum((x - mean) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)

    # IQR
    q1 = vals[n // 4]
    q3 = vals[(3 * n) // 4]
    iqr = q3 - q1

    return {
        "median": median,
        "mean": mean,
        "std": std,
        "iqr": iqr,
        "min": vals[0],
        "max": vals[-1],
        "count": n
    }


def analyze_benchmarks(json_path: Path, output_dir: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    context = data.get("context", {})
    benchmarks = data.get("benchmarks", [])

    # Group runs by benchmark configuration
    grouped_runs = defaultdict(lambda: {"real_times": [], "cpu_times": [], "bytes_per_second": [], "items_per_second": []})

    for b in benchmarks:
        name = b["name"]
        # Filter out Google Benchmark aggregate summary rows
        if any(name.endswith(f"_{agg}") for agg in ["mean", "median", "stddev", "cv"]):
            continue

        info = parse_benchmark_name(name)
        key = info["clean_name"]
        grouped_runs[key]["info"] = info
        grouped_runs[key]["real_times"].append(b["real_time"])
        grouped_runs[key]["cpu_times"].append(b["cpu_time"])
        if "bytes_per_second" in b and b["bytes_per_second"] > 0:
            grouped_runs[key]["bytes_per_second"].append(b["bytes_per_second"])
        if "items_per_second" in b and b["items_per_second"] > 0:
            grouped_runs[key]["items_per_second"].append(b["items_per_second"])

    summary_metrics = {}
    for key, group in grouped_runs.items():
        summary_metrics[key] = {
            "info": group["info"],
            "real_time": compute_statistics(group["real_times"]),
            "cpu_time": compute_statistics(group["cpu_times"]),
            "bandwidth_gb_per_sec": compute_statistics([x / 1e9 for x in group["bytes_per_second"]]),
            "throughput_mit_per_sec": compute_statistics([x / 1e6 for x in group["items_per_second"]]),
        }

    # Speedup and efficiency calculations
    speedup_results = {}
    # Find parallel benchmarks with thread parameter (typically 2nd arg)
    for key, stats in summary_metrics.items():
        info = stats["info"]
        args = info["args"]
        if "Parallel" in info["kernel_tag"] and len(args) >= 2:
            workload_size = args[0]
            threads = args[1]
            base_key = f"{info['kernel_tag']}/{workload_size}/1"
            if base_key in summary_metrics:
                t1 = summary_metrics[base_key]["real_time"]["median"]
                tp = stats["real_time"]["median"]
                speedup = t1 / tp if tp > 0 else 0.0
                efficiency = (speedup / threads) if threads > 0 else 0.0
                speedup_results[key] = {
                    "kernel": info["kernel_tag"],
                    "workload_size": workload_size,
                    "threads": threads,
                    "t1_ns": t1,
                    "tp_ns": tp,
                    "speedup": speedup,
                    "efficiency": efficiency
                }

    # Export summary JSON
    out_json = output_dir / "summary_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "context": context,
            "metrics": summary_metrics,
            "speedup_efficiency": speedup_results
        }, f, indent=2)
    print(f"Summary metrics exported to: {out_json}")

    # Generate Markdown Summary Table
    md_path = output_dir / "summary_table.md"
    generate_markdown_report(context, summary_metrics, speedup_results, md_path)
    print(f"Markdown summary report exported to: {md_path}")

    # Generate Plots if matplotlib available
    if HAS_PLOT_LIBS:
        generate_plots(summary_metrics, speedup_results, output_dir)
    else:
        print("[Warning] Matplotlib not available; skipping plot generation.")


def generate_markdown_report(context: dict, metrics: dict, speedup_results: dict, out_path: Path):
    lines = [
        "# ParallelComputeLab Experimental Results Summary",
        "",
        "## 1. Environment Provenance",
        f"- **CPU Model**: {context.get('cpu_model', 'Unknown')}",
        f"- **Logical Cores**: {context.get('logical_cores', 'Unknown')}",
        f"- **Physical Cores**: {context.get('physical_cores', 'Unknown')}",
        f"- **Caches**: L1d: {int(context.get('l1d_cache_bytes', 0)) // 1024} KiB, "
        f"L2: {int(context.get('l2_cache_bytes', 0)) // 1024} KiB, "
        f"L3: {int(context.get('l3_cache_bytes', 0)) // (1024 * 1024)} MiB",
        f"- **Compiler**: {context.get('compiler_id', 'Unknown')} {context.get('compiler_version', '')} ({context.get('build_type', 'Unknown')})",
        f"- **SIMD Features**: AVX2: {context.get('has_avx2', 'false')}, FMA: {context.get('has_fma', 'false')}",
        "",
        "## 2. Kernel Scaling, Speedup, and Parallel Efficiency",
        "",
        "| Kernel | Workload Size | Threads | Median Time (ns) | Speedup S(p) | Efficiency E(p) |",
        "|---|---|---|---|---|---|"
    ]

    for key, sp in sorted(speedup_results.items(), key=lambda x: (x[1]['kernel'], x[1]['workload_size'], x[1]['threads'])):
        lines.append(
            f"| `{sp['kernel']}` | {sp['workload_size']} | {sp['threads']} | "
            f"{sp['tp_ns']:.2f} | **{sp['speedup']:.2f}x** | {sp['efficiency'] * 100:.1f}% |"
        )

    lines.extend([
        "",
        "## 3. Implementation Variant Comparison (Scalar vs Optimized vs SIMD)",
        "",
        "| Workload / Size | Variant | Median Latency (ns) | IQR (ns) | Bandwidth / Throughput |",
        "|---|---|---|---|---|"
    ])

    for key, item in sorted(metrics.items()):
        info = item["info"]
        tag = info["kernel_tag"]
        if "Parallel" not in tag:
            size_str = str(info["args"][0]) if info["args"] else "-"
            bw = item["bandwidth_gb_per_sec"]["median"]
            bw_str = f"{bw:.2f} GB/s" if bw > 0 else "-"
            lines.append(
                f"| `{tag}` ({size_str}) | `{tag.split('_')[-1]}` | "
                f"{item['real_time']['median']:.2f} | {item['real_time']['iqr']:.2f} | {bw_str} |"
            )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_plots(metrics: dict, speedup_results: dict, output_dir: Path):
    plt.style.use('tableau-colorblind10')

    # 1. Thread Scaling Plot (Speedup & Efficiency)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Group by (kernel, size)
    scaling_series = defaultdict(lambda: {"threads": [], "speedup": [], "efficiency": []})
    for _, sp in speedup_results.items():
        series_label = f"{sp['kernel'].replace('BM_', '')} (N={sp['workload_size']})"
        scaling_series[series_label]["threads"].append(sp["threads"])
        scaling_series[series_label]["speedup"].append(sp["speedup"])
        scaling_series[series_label]["efficiency"].append(sp["efficiency"])

    for label, vals in scaling_series.items():
        order = np.argsort(vals["threads"])
        th = np.array(vals["threads"])[order]
        sp = np.array(vals["speedup"])[order]
        ef = np.array(vals["efficiency"])[order]

        ax1.plot(th, sp, marker='o', linewidth=2, label=label)
        ax2.plot(th, ef * 100, marker='s', linewidth=2, label=label)

    # Ideal linear speedup line
    max_th = 8
    ax1.plot([1, max_th], [1, max_th], 'k--', alpha=0.5, label='Ideal linear')
    ax1.set_xlabel("Thread Count p")
    ax1.set_ylabel("Speedup S(p) = T(1) / T(p)")
    ax1.set_title("Thread Scaling Speedup")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=8)

    ax2.axhline(100, color='k', linestyle='--', alpha=0.5, label='100% ideal')
    ax2.set_xlabel("Thread Count p")
    ax2.set_ylabel("Parallel Efficiency E(p) (%)")
    ax2.set_title("Parallel Scaling Efficiency")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    plot_path = output_dir / "scaling_threads.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"Plot saved: {plot_path}")

    # 2. Memory Strided Access Degradation Plot
    strided_points = []
    for key, item in metrics.items():
        info = item["info"]
        if info["kernel_tag"] == "BM_Memory_StridedRead" and len(info["args"]) >= 2:
            stride = info["args"][1]
            bw = item["bandwidth_gb_per_sec"]["median"]
            lat = item["real_time"]["median"]
            strided_points.append((stride, bw, lat))

    if strided_points:
        strided_points.sort(key=lambda x: x[0])
        strides = [p[0] for p in strided_points]
        lats = [p[2] for p in strided_points]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([str(s) for s in strides], lats, marker='o', color='#d62728', linewidth=2)
        ax.set_xlabel("Stride (elements)")
        ax.set_ylabel("Latency per Traversal (ns)")
        ax.set_title("Memory Access Penalty vs Stride (Cache Line Eviction Effect)")
        ax.grid(True, linestyle='--', alpha=0.6)

        fig.tight_layout()
        stride_plot = output_dir / "memory_bandwidth.png"
        fig.savefig(stride_plot, dpi=200)
        plt.close(fig)
        print(f"Plot saved: {stride_plot}")

    # 3. Workload Comparison: Scalar vs Optimized vs SIMD vs Parallel
    variants = ["Scalar", "Optimized", "SIMD", "Parallel"]
    target_sizes = [1024, 16384, 262144, 4194304]
    va_data = defaultdict(dict)

    for v in variants:
        for sz in target_sizes:
            k = f"BM_VectorAdd_{v}/{sz}" if v != "Parallel" else f"BM_VectorAdd_Parallel/{sz}/4"
            if k in metrics:
                va_data[v][sz] = metrics[k]["real_time"]["median"]

    if all(v in va_data and len(va_data[v]) > 0 for v in ["Scalar", "Optimized", "SIMD"]):
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(target_sizes))
        width = 0.2

        for i, v in enumerate(variants):
            vals = [va_data[v].get(sz, 0) / 1000.0 for sz in target_sizes] # in microseconds
            ax.bar(x + (i - 1.5) * width, vals, width, label=v)

        ax.set_xticks(x)
        ax.set_xticklabels(["1K (L1)", "16K (L2)", "256K (L3)", "4M (DRAM)"])
        ax.set_xlabel("Working Set Size (elements)")
        ax.set_ylabel("Execution Time (µs) - log scale")
        ax.set_yscale("log")
        ax.set_title("Vector Addition: Optimization Levels Across Working Set Sizes")
        ax.grid(True, linestyle='--', alpha=0.5, which="both")
        ax.legend()

        fig.tight_layout()
        workload_plot = output_dir / "workload_comparison.png"
        fig.savefig(workload_plot, dpi=200)
        plt.close(fig)
        print(f"Plot saved: {workload_plot}")

    # 4. OpenMP Scheduling Policy Comparison
    sched_data = defaultdict(dict)
    for key, item in metrics.items():
        info = item["info"]
        if "Schedule" in info["kernel_tag"] and len(info["args"]) >= 4:
            workload = "Uniform" if "Uniform" in info["kernel_tag"] else "Imbalanced"
            policy_idx = info["args"][1]
            chunk = info["args"][2]
            pol_map = {0: "Serial", 1: f"Static({chunk})", 2: f"Dynamic({chunk})", 3: f"Guided({chunk})"}
            pol_name = pol_map.get(policy_idx, f"Policy_{policy_idx}")
            sched_data[workload][pol_name] = item["real_time"]["median"] / 1e6 # in ms

    if sched_data:
        fig, ax = plt.subplots(figsize=(9, 5))
        policies = list(dict.fromkeys([p for w in sched_data.values() for p in w.keys()]))
        x = np.arange(len(policies))
        width = 0.35

        for i, wl in enumerate(["Uniform", "Imbalanced"]):
            vals = [sched_data[wl].get(p, 0) for p in policies]
            ax.bar(x + (i - 0.5) * width, vals, width, label=f"{wl} Workload")

        ax.set_xticks(x)
        ax.set_xticklabels(policies, rotation=15)
        ax.set_ylabel("Runtime (ms)")
        ax.set_title("OpenMP Scheduling Policies: Uniform vs Imbalanced Workloads")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        fig.tight_layout()
        sched_plot = output_dir / "scheduling_comparison.png"
        fig.savefig(sched_plot, dpi=200)
        plt.close(fig)
        print(f"Plot saved: {sched_plot}")


def main():
    if len(sys.argv) < 2:
        json_path = Path("results/raw_benchmark_results.json")
    else:
        json_path = Path(sys.argv[1])

    output_dir = json_path.parent
    if len(sys.argv) >= 3:
        output_dir = Path(sys.argv[2])

    if not json_path.exists():
        # Try alternate location
        alt_path = Path("ParallelComputeLab") / json_path
        if alt_path.exists():
            json_path = alt_path
            output_dir = json_path.parent
        else:
            print(f"Error: JSON file not found at {json_path}", file=sys.stderr)
            sys.exit(1)

    analyze_benchmarks(json_path, output_dir)


if __name__ == "__main__":
    main()
