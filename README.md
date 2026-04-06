# MLX Vision-Language Model Benchmark

Benchmarking **Gemma 4** and **Qwen 3.5** models on Apple Silicon using [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), measuring prefill and decode throughput across context lengths from 4k to 256k tokens.

Includes the effect of **TurboQuant** — mlx-vlm's KV cache quantization (2.5-bit) — which provides significant decode speedups at long contexts with zero memory overhead.

## Results

**[View the interactive HTML report with charts](results/benchmark_report.html)**

Full tables: [results/BENCHMARK_REPORT.md](results/BENCHMARK_REPORT.md)

Raw data: [results/bench_all_results.json](results/bench_all_results.json)

### Models tested

| Model | Type | Quantisations | 4k Decode (4bit) | 256k Decode (4bit) | Mem (4bit, 4k) |
|-------|------|---------------|------------------|--------------------|----------------|
| **Gemma 4 E2B** | MoE, ~2B active | 4bit, 8bit, bf16 | 205 t/s | 78 t/s | 4.7 GB |
| **Gemma 4 E4B** | MoE, ~4B active | 4bit, 8bit, bf16 | 127 t/s | 27 t/s | 6.4 GB |
| **Gemma 4 26B-A4B** | MoE, ~4B active, 26B total | 4bit, 8bit, bf16 | 113 t/s | 30 t/s | 17.1 GB |
| **Gemma 4 31B** | Dense, 31B | 4bit | 27 t/s | 7 t/s | 22.7 GB |
| **Qwen 3.5 2B** | Dense, 2B | 4bit, 8bit, bf16 | 308 t/s | 61 t/s | 3.3 GB |
| **Qwen 3.5 4B** | Dense, 4B | 4bit, 8bit, bf16 | 156 t/s | 29 t/s | 5.2 GB |

### TurboQuant highlights

TurboQuant quantizes the KV cache to 2.5 bits during inference, reducing memory bandwidth during decoding. The benefit scales with context length:

- **Negligible at short contexts** (< 16k) — KV cache is small, overhead outweighs savings
- **+5-10% at 32-64k** — crossover point where bandwidth savings start to dominate
- **+15-19% at 128-256k** — substantial free speedup on larger models (26B, 31B, E4B)
- **Negative on very small models** (E2B) — model weights are so small that the quantization overhead dominates

## Hardware

Results below were collected on:

- **Apple M5 Max** — 18 cores (6 Super + 12 Performance), 128 GB unified memory
- **macOS** Darwin 25.4.0
- **MLX** 0.31.1, **mlx-vlm** 0.4.4

## Running the benchmark

### Prerequisites

```bash
pip install mlx-vlm pyyaml
```

Models should be downloaded to `~/.lmstudio/models/` (or update `models_base_dir` in `config.yaml`).

### Configuration

Edit `config.yaml` to configure:
- `models_base_dir` — where your MLX models live
- `models` — which models to benchmark (name, family, path)
- `context_targets` — context lengths to test
- `turboquant_kv_bits` — KV cache quantization bits (2.5 recommended)
- `families` — per-family sampling parameters (temperature, top_p, enable_thinking)

### Run all benchmarks

```bash
python3 bench_run.py
```

Each model runs in a separate subprocess for accurate peak memory reporting. Results are saved to `results/bench_all_results.json`.

### Run a single model

```bash
python3 bench_one_model.py "Model Name" family_id /path/to/model
```

Outputs JSON lines to stdout — one per context length.

### Generate reports

```bash
python3 generate_reports.py
```

Produces `results/BENCHMARK_REPORT.md` and `results/benchmark_report.html` from the JSON results.

## Project structure

```
config.yaml              # Model paths, parameters, report config
bench_run.py             # Orchestrator — runs all models via subprocesses
bench_one_model.py       # Single-model benchmark (subprocess worker)
generate_reports.py      # Generates markdown + HTML reports from results
results/
  bench_all_results.json # Raw benchmark data
  BENCHMARK_REPORT.md    # Full results tables
  benchmark_report.html  # Interactive report with Chart.js graphs
```

## License

MIT
