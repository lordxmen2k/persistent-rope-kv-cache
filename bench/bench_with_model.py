"""Real-model benchmark: distilgpt2 + PersistentRope as shadow cache.

Runs HuggingFace generation while tracking the KV cache in parallel with
the rope. The rope acts as a 'shadow' index; we measure the rope's
operations during real model inference.

Auto-detects CUDA and runs on GPU if available. Falls back to CPU.

Run:
    python -m bench.bench_with_model                  # auto-detect device
    python -m bench.bench_with_model --device cuda    # force GPU
    python -m bench.bench_with_model --device cpu     # force CPU
"""
import sys
import time
import statistics
import argparse

sys.path.insert(0, "src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rope import RopeBlockManager, Tier


def detect_device(forced: str = None) -> str:
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(model_name: str = "distilgpt2", max_new_tokens: int = 50,
         n_prompts: int = 8, device: str = None):
    device = detect_device(device)
    print("=" * 60)
    print(f"Real-model benchmark: {model_name} on {device.upper()}")
    print("=" * 60)

    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading {model_name}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    t_load = time.perf_counter() - t0
    print(f"  loaded in {t_load:.1f}s ({n_params/1e6:.1f}M params)")

    # Prompts of varying length
    base_prompt = "The history of artificial intelligence is a long and fascinating story. "
    prompts = [
        base_prompt + "It involves the development of computational systems that can perform tasks."
        for _ in range(n_prompts)
    ]
    # Add prefix-sharing: half the prompts share a long system prompt
    shared = "You are a helpful assistant. Please answer the following question carefully and accurately. "
    prompts_with_share = [shared + p for p in prompts[:n_prompts // 2]] + prompts[n_prompts // 2:]

    print(f"\nGenerating {max_new_tokens} tokens for {n_prompts} prompts...")
    bm = RopeBlockManager(
        block_size=16,
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=getattr(model.config, "num_key_value_heads",
                            getattr(model.config, "num_attention_heads", 12)),
        head_dim=getattr(model.config, "head_dim",
                         model.config.hidden_size // model.config.num_attention_heads),
    )

    # Track per-sequence timing
    seq_timings = []
    seq_lengths = []

    if device == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for i, prompt in enumerate(prompts_with_share):
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            prompt_len = input_ids.shape[1]

            # Create sequence in rope block manager
            bm.create_sequence(i + 1)
            t_rope_setup = time.perf_counter()
            bm.allocate(i + 1, n_tokens=prompt_len, kv_data=b"p" * prompt_len)
            t_rope_setup = time.perf_counter() - t_rope_setup

            # Generate
            if device == "cuda":
                torch.cuda.synchronize()
            t_model = time.perf_counter()
            out_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            t_model = time.perf_counter() - t_model
            new_tokens = out_ids.shape[1] - prompt_len

            # Track new tokens in rope
            t_rope_emit = time.perf_counter()
            for _ in range(new_tokens):
                bm.append_slot(i + 1, kv_data=b"x")
            t_rope_emit = time.perf_counter() - t_rope_emit

            # Verify rope snapshot matches actual cache
            actual_len = prompt_len + new_tokens
            snap = bm.get_snapshot(i + 1)
            rope_len = sum(s[2] for s in snap)
            if rope_len != actual_len:
                print(f"  seq {i+1}: MISMATCH actual={actual_len} rope={rope_len}")

            seq_timings.append({
                "prompt_len": prompt_len,
                "new_tokens": new_tokens,
                "t_rope_setup": t_rope_setup,
                "t_model": t_model,
                "t_rope_emit": t_rope_emit,
            })
            seq_lengths.append(actual_len)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\n--- per-sequence timings ---")
    print(f"  {'seq':<4} {'prompt':<7} {'new':<5} {'t_model':<10} {'t_rope':<12} {'tok/s':<8}")
    for i, t in enumerate(seq_timings):
        tok_per_sec = t["new_tokens"] / t["t_model"] if t["t_model"] > 0 else 0
        rope_total = t["t_rope_setup"] + t["t_rope_emit"]
        print(f"  {i+1:<4} {t['prompt_len']:<7} {t['new_tokens']:<5} "
              f"{t['t_model']*1000:<10.1f} {rope_total*1000:<12.2f} {tok_per_sec:<8.2f}")

    print(f"\n--- aggregate ---")
    total_model_t = sum(t["t_model"] for t in seq_timings)
    total_rope_t = sum(t["t_rope_setup"] + t["t_rope_emit"] for t in seq_timings)
    total_new = sum(t["new_tokens"] for t in seq_timings)
    total_prompt = sum(t["prompt_len"] for t in seq_timings)
    print(f"  total sequences:       {len(seq_timings)}")
    print(f"  total prompt tokens:   {total_prompt}")
    print(f"  total new tokens:      {total_new}")
    print(f"  total model time:      {total_model_t:.3f}s")
    print(f"  total rope time:       {total_rope_t*1000:.2f}ms")
    print(f"  rope / model:          {total_rope_t / total_model_t * 100:.4f}%")
    print(f"  aggregate tok/s:       {total_new / total_model_t:.2f}")

    # Per-op rope cost
    print(f"\n--- per-op rope cost ---")
    rope_per_op_setup = statistics.mean([t["t_rope_setup"] / t["prompt_len"]
                                          for t in seq_timings if t["prompt_len"] > 0]) * 1e6
    rope_per_op_emit = statistics.mean([t["t_rope_emit"] / t["new_tokens"]
                                         for t in seq_timings if t["new_tokens"] > 0]) * 1e6
    print(f"  per-token rope setup:  {rope_per_op_setup:.2f} us")
    print(f"  per-token rope emit:   {rope_per_op_emit:.2f} us")

    # Multi-tenant summary
    print(f"\n--- multi-tenant snapshot ---")
    s = bm.stats()
    print(f"  n_sequences in rope:   {s['n_sequences']}")
    print(f"  rope leaves:           {s['rope_stats']['n_leaves']}")
    print(f"  rope internals:        {s['rope_stats']['n_internals']}")
    print(f"  mapping size:          {s['rope_stats']['n_nodes_total']}")
    print(f"  rope length (sum):     {s['rope_stats']['length']}")

    print(f"\n--- rope overhead analysis ---")
    print(f"  total tokens tracked by rope:  {sum(seq_lengths)}")
    print(f"  rope bookkeeping:              {total_rope_t*1000:.2f}ms")
    print(f"  rope per token:                 {total_rope_t / sum(seq_lengths) * 1e6:.2f} us")
    print(f"  model attention:                {total_model_t*1000:.1f}ms")
    print(f"  rope as % of model cost:        {total_rope_t / total_model_t * 100:.4f}%")

    return {
        "device": device,
        "model": model_name,
        "n_seqs": len(seq_timings),
        "total_prompt": total_prompt,
        "total_new": total_new,
        "total_model_s": total_model_t,
        "total_rope_ms": total_rope_t * 1000,
        "rope_pct": total_rope_t / total_model_t * 100,
        "tok_per_sec": total_new / total_model_t,
        "rope_per_token_us": total_rope_t / sum(seq_lengths) * 1e6,
        "rope_per_op_setup_us": rope_per_op_setup,
        "rope_per_op_emit_us": rope_per_op_emit,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detect if not set)")
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--n-prompts", type=int, default=8)
    args = parser.parse_args()
    main(model_name=args.model, max_new_tokens=args.max_new_tokens,
         n_prompts=args.n_prompts, device=args.device)
