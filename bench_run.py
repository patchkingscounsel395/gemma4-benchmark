#!/usr/bin/env python3
"""
Run all benchmarks defined in config.yaml.
Each model runs in a separate subprocess for accurate memory reporting.
Results are saved to the configured results file.
"""

import glob
import json
import os
import subprocess
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

MODELS_BASE = os.path.expanduser(CONFIG["models_base_dir"])
RESULTS_FILE = os.path.join(SCRIPT_DIR, CONFIG["results_file"])


def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    all_results = []

    for model_def in CONFIG["models"]:
        display_name = model_def["name"]
        family = model_def["family"]
        path = os.path.join(MODELS_BASE, model_def["path"])

        # Check model exists
        if not os.path.exists(os.path.join(path, "config.json")):
            print(f"  [{display_name}] NOT FOUND at {path} — skipping", file=sys.stderr, flush=True)
            continue
        safetensors = glob.glob(os.path.join(path, "*.safetensors"))
        if not safetensors:
            print(f"  [{display_name}] No weights at {path} — skipping", file=sys.stderr, flush=True)
            continue

        print(f"\n{'='*60}", file=sys.stderr, flush=True)
        print(f"  {display_name}", file=sys.stderr, flush=True)
        print(f"{'='*60}", file=sys.stderr, flush=True)

        proc = subprocess.run(
            ["python3", "-u", os.path.join(SCRIPT_DIR, "bench_one_model.py"), display_name, family, path],
            capture_output=True, text=True, timeout=7200,
        )

        for line in proc.stdout.strip().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            all_results.append(row)
            ctx = row["target_ctx"] // 1000
            print(f"  {ctx:>5}k  std: {row.get('prefill',0):>7.0f}/{row.get('decode',0):>5.1f} t/s {row.get('mem',0):>5.1f}GB  "
                  f"TQ: {row.get('tq_prefill',0):>7.0f}/{row.get('tq_decode',0):>5.1f} t/s {row.get('tq_mem',0):>5.1f}GB",
                  file=sys.stderr, flush=True)

        if proc.returncode != 0:
            print(f"  ERROR: {proc.stderr[-500:]}", file=sys.stderr, flush=True)

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved {len(all_results)} rows to {RESULTS_FILE}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
