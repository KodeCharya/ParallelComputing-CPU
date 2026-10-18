#!/usr/bin/env python3
"""
collect_system_info.py
Collects comprehensive host system, CPU, cache topology, compiler, and git metadata.
Outputs structured JSON conforming to the ParallelComputeLab metadata schema.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def get_git_commit(cwd: Path) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_cli_diagnostics(binary_path: Path) -> dict:
    if binary_path.exists() and os.access(binary_path, os.X_OK):
        try:
            res = subprocess.run(
                [str(binary_path), "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            return json.loads(res.stdout)
        except Exception as e:
            print(f"[Warning] Could not parse CLI diagnostic output: {e}", file=sys.stderr)
    return {}


def collect_system_info(repo_root: Path, output_file: Path | None = None) -> dict:
    candidates = [
        repo_root / "build" / "release" / "system_info_cli",
        repo_root.parent / "build" / "release" / "system_info_cli",
        repo_root / "ParallelComputeLab" / "build" / "release" / "system_info_cli",
        Path("build/release/system_info_cli"),
    ]
    cli_binary = None
    for c in candidates:
        if c.exists():
            cli_binary = c
            break

    cli_data = get_cli_diagnostics(cli_binary) if cli_binary else {}

    info = {
        "schema_version": "1.0.0",
        "timestamp_utc": subprocess.getoutput("date -u +'%Y-%m-%dT%H:%M:%SZ'"),
        "git_commit": get_git_commit(repo_root),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform()
        },
        "python_version": platform.python_version(),
        "cpu": {
            "model": cli_data.get("cpu_model", platform.processor()),
            "vendor": cli_data.get("cpu_vendor", "Unknown"),
            "logical_cores": cli_data.get("logical_cores", os.cpu_count() or 1),
            "physical_cores": cli_data.get("physical_cores", max(1, (os.cpu_count() or 1) // 2)),
            "openmp_max_threads": cli_data.get("openmp_max_threads", os.cpu_count() or 1),
            "cache": cli_data.get("cache", {
                "l1d_bytes": 32768,
                "l1i_bytes": 32768,
                "l2_bytes": 262144,
                "l3_bytes": 6291456
            }),
            "features": cli_data.get("features", {})
        },
        "build": cli_data.get("build", {
            "compiler_id": "GNU",
            "compiler_version": "14.2.0",
            "build_type": "Release"
        })
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(f"System metadata written to: {output_file}")

    return info


def main():
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "results" / "system_metadata.json"
    if len(sys.argv) > 1:
        output_path = Path(sys.argv[1])

    info = collect_system_info(repo_root, output_path)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
