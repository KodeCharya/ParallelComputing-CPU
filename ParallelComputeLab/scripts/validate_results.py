#!/usr/bin/env python3
"""
validate_results.py
Validates the structural integrity and numeric validity of benchmark output files.
Rejects missing, corrupted, negative, or NaN measurements.
"""

import json
import math
import sys
from pathlib import Path


def validate_benchmark_json(file_path: Path) -> bool:
    if not file_path.exists():
        print(f"Error: Results file not found: {file_path}", file=sys.stderr)
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to parse JSON: {e}", file=sys.stderr)
        return False

    # Check top-level fields
    if "context" not in data:
        print("Validation Error: Missing 'context' object in benchmark output.", file=sys.stderr)
        return False

    context = data["context"]
    required_context = ["cpu_model", "compiler_id", "os_name"]
    for req in required_context:
        if req not in context or not context[req]:
            print(f"Validation Error: Missing required context field '{req}'.", file=sys.stderr)
            return False

    if "benchmarks" not in data or not isinstance(data["benchmarks"], list):
        print("Validation Error: Missing or invalid 'benchmarks' array.", file=sys.stderr)
        return False

    benchmarks = data["benchmarks"]
    if len(benchmarks) == 0:
        print("Validation Error: 'benchmarks' array is empty.", file=sys.stderr)
        return False

    required_bench_fields = ["name", "real_time", "cpu_time", "iterations", "time_unit"]
    valid_count = 0
    for idx, bench in enumerate(benchmarks):
        for field in required_bench_fields:
            if field not in bench:
                print(f"Validation Error: Benchmark entry #{idx} missing field '{field}'.", file=sys.stderr)
                return False

        # Numeric validations
        for time_field in ["real_time", "cpu_time"]:
            val = bench[time_field]
            if not isinstance(val, (int, float)):
                print(f"Validation Error: Entry #{idx} '{bench['name']}' field '{time_field}' is not numeric.", file=sys.stderr)
                return False
            if math.isnan(val) or math.isinf(val) or val <= 0:
                print(f"Validation Error: Entry #{idx} '{bench['name']}' has invalid {time_field}={val}.", file=sys.stderr)
                return False

        iters = bench["iterations"]
        if not isinstance(iters, int) or iters <= 0:
            print(f"Validation Error: Entry #{idx} '{bench['name']}' has invalid iterations={iters}.", file=sys.stderr)
            return False

        valid_count += 1

    print(f"Validation SUCCESS: {valid_count} benchmark entries in {file_path} are valid.")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 validate_results.py <path_to_benchmark_results.json>")
        sys.exit(1)

    target_file = Path(sys.argv[1])
    is_valid = validate_benchmark_json(target_file)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
