#!/usr/bin/env python3
"""Benchmark a single model, output JSON lines to stdout.
Run in a subprocess for accurate peak memory reporting."""

import json
import os
import sys

import yaml
import mlx.core as mx
from mlx_vlm import load, generate

FILLER = "The quick brown fox jumps over the lazy dog. "

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

TARGETS = CONFIG["context_targets"]
TQ_BITS = CONFIG.get("turboquant_kv_bits", 2.5)
FAMILIES = CONFIG["families"]


def run_one(model, processor, tokenizer, family, target_tokens, kv_bits=None):
    filler_block = FILLER * 200
    chars_per_token = len(filler_block) / len(tokenizer.encode(filler_block))
    needed_chars = int(target_tokens * chars_per_token)
    padding = (FILLER * (needed_chars // len(FILLER) + 1))[:needed_chars]
    messages = [{"role": "user", "content": padding + "\n\nSummarize the above in one sentence."}]

    fam = FAMILIES[family]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=fam.get("enable_thinking", False),
    )
    actual = len(tokenizer.encode(prompt))

    kwargs = {"max_tokens": 32, "temp": fam["temperature"], "top_p": fam["top_p"], "verbose": False}
    if kv_bits is not None:
        kwargs["kv_bits"] = kv_bits

    result = generate(model, processor, prompt, **kwargs)
    return actual, result.prompt_tps, result.generation_tps, result.peak_memory


def main():
    display_name = sys.argv[1]
    family = sys.argv[2]
    path = sys.argv[3]

    model, processor = load(path)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Warmup
    fam = FAMILIES[family]
    p = processor.apply_chat_template(
        [{"role": "user", "content": "Hi"}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=fam.get("enable_thinking", False),
    )
    _ = generate(model, processor, p, max_tokens=5, verbose=False)

    for target in TARGETS:
        row = {"model": display_name, "family": family, "target_ctx": target}
        try:
            actual, prefill, decode, mem = run_one(model, processor, tokenizer, family, target)
            row.update({"actual_ctx": actual, "prefill": round(prefill, 1), "decode": round(decode, 1), "mem": round(mem, 1)})
        except Exception as e:
            row.update({"prefill": 0, "decode": 0, "mem": 0, "error": str(e)})

        try:
            _, tq_prefill, tq_decode, tq_mem = run_one(model, processor, tokenizer, family, target, kv_bits=TQ_BITS)
            row.update({"tq_prefill": round(tq_prefill, 1), "tq_decode": round(tq_decode, 1), "tq_mem": round(tq_mem, 1)})
        except Exception as e:
            row.update({"tq_prefill": 0, "tq_decode": 0, "tq_mem": 0, "tq_error": str(e)})

        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
