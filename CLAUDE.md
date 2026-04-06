# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MLX benchmark suite for vision-language models (Gemma 4, Qwen 3.5) on Apple Silicon. Measures prefill and decode throughput across context lengths (4k-256k) with and without TurboQuant KV cache quantization (2.5-bit). All models use **mlx-vlm** (not mlx-lm) because they are vision-language models.

## Commands

```bash
# Run full benchmark (all models in config.yaml, each in a subprocess)
python3 bench_run.py

# Run a single model
python3 bench_one_model.py "Model Name" family_id /path/to/model

# Regenerate reports from existing results
python3 generate_reports.py

# Install dependencies
pip install mlx-vlm pyyaml
```

## Architecture

**`config.yaml`** is the single source of truth for model paths, sampling parameters, context targets, and report groupings. All scripts read from it. Model paths are relative to `models_base_dir` (default `~/.lmstudio/models`).

**`bench_run.py`** orchestrates benchmarks by spawning `bench_one_model.py` as a **separate subprocess per model**. This is critical — mlx's `peak_memory` reports process-level peak and never decreases after `del model`. Without subprocess isolation, memory figures accumulate and are wrong.

**`bench_one_model.py`** loads one model via `mlx_vlm.load()`, runs all context lengths (standard + TurboQuant), outputs JSON lines to stdout. The orchestrator collects these.

**`generate_reports.py`** reads `results/bench_all_results.json` and produces both markdown tables and an HTML report with Chart.js line graphs (log-scale x-axis for context, decode t/s on y).

## Key technical details

- **Thinking mode must be OFF** for benchmarks: pass `enable_thinking=False` to `apply_chat_template()`. Gemma 4 defaults to off; Qwen 3.5 4B defaults to ON, so it must be explicitly disabled.
- **Sampling params differ by family**: Gemma 4 uses temp=1.0/top_p=0.95 (per generation_config.json); Qwen 3.5 uses temp=0.7/top_p=0.8 (per model card, non-thinking mode).
- **TurboQuant** is enabled by passing `kv_bits=2.5` to `mlx_vlm.generate()`. It quantizes the KV cache during inference. Uses fractional bits (2.5) which triggers TurboQuant mode automatically in mlx-vlm (vs integer bits which use uniform quantization).
- Results JSON uses `target_ctx` (requested) which maps to standard sizes. The `actual_ctx` field has the real token count after tokenization.

## Important context

- The `app.py` and `templates/` directory contain a separate Flask web UI for interactive model testing (not part of the benchmark). They are gitignored.
- The old benchmark scripts (`bench_all.py`, `bench_rerun_mem.py`, `bench_turboquant.py`, `bench_26b_8bit.py`) are superseded by `bench_run.py` + `bench_one_model.py` and are gitignored.
- Hardware is Apple M5 Max with 128 GB unified memory. This matters for which models fit — the 31B bf16 at 256k needs ~81 GB.
