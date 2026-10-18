#!/usr/bin/env python3
"""
run_benchmarks.py
Executes the ParallelComputeLab benchmark suite with user-specified filters,
repetitions, and timing controls. Saves raw JSON and CSV results and system metadata.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from collect_system_info import collect_system_info
from validate_results import validate_benchmark_json


def run_benchmarks(
    binary: Path,
    output_dir: Path,
    benchmark_filter: str | None = None,
    repetitions: int = 5,
    min_time: float = 0.2
):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "raw_benchmark_results.json"
    csv_path = output_dir / "raw_benchmark_results.csv"
    meta_path = output_dir / "system_metadata.json"

    print("[1/3] Collecting host and hardware metadata...")
    repo_root = binary.resolve().parent.parent.parent
    collect_system_info(repo_root, meta_path)

    cmd = [
        str(binary),
        f"--benchmark_out={json_path}",
        "--benchmark_out_format=json",
        f"--benchmark_repetitions={repetitions}",
        "--benchmark_report_aggregates_only=false",
        f"--benchmark_min_time={min_time}s"
    ]

    if benchmark_filter:
        cmd.append(f"--benchmark_filter={benchmark_filter}")

    print(f"[2/3] Executing benchmarks: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Error: Benchmark binary exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    # Also export CSV format if possible
    csv_cmd = [
        str(binary),
        f"--benchmark_out={csv_path}",
        "--benchmark_out_format=csv",
        "--benchmark_repetitions=1",
        f"--benchmark_min_time={min_time}s"
    ]
    if benchmark_filter:
        csv_cmd.append(f"--benchmark_filter={benchmark_filter}")
    subprocess.run(csv_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[3/3] Validating benchmark output...")
    if not validate_benchmark_json(json_path):
        print("Validation FAILED on generated benchmark data!", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll benchmarks completed successfully.")
    print(f"JSON results: {json_path}")
    print(f"CSV results:  {csv_path}")
    print(f"Metadata:     {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Run ParallelComputeLab benchmarks")
    parser.add_argument("--binary", type=Path, default=Path("build/release/parallel_compute_benchmarks"),
                        help="Path to parallel_compute_benchmarks binary")
    parser.add_argument("--output-dir", type=Path, default=Path("results"),
                        help="Directory to save raw results")
    parser.add_argument("--filter", type=str, default=None,
                        help="Regex filter for benchmark names")
    parser.add_argument("--repetitions", type=int, default=3,
                        help="Number of repetitions per benchmark")
    parser.add_argument("--min-time", type=float, default=0.2,
                        help="Minimum runtime per repetition in seconds")

    args = parser.parse_args()

    binary_path = args.binary
    if not binary_path.exists():
        # Try finding in ParallelComputeLab subfolder
        alt_path = Path("ParallelComputeLab") / args.binary
        if alt_path.exists():
            binary_path = alt_path
        else:
            print(f"Error: Benchmark executable not found at {args.binary}", file=sys.stderr)
            sys.exit(1)

    run_benchmarks(
        binary=binary_path,
        output_dir=args.output_dir,
        benchmark_filter=args.filter,
        repetitions=args.repetitions,
        min_time=args.min_time
    )


if __name__ == "__main__":
    main()
