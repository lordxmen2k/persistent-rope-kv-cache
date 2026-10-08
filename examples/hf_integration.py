"""HuggingFace integration demo: use the rope as a logical KV cache tracker.

This script loads a small HuggingFace model (default: distilgpt2) and runs
inference, tracking the KV cache state in parallel using the
PersistentRope. The rope acts as a "shadow" of HF's past_key_values,
demonstrating the API surface the vLLM-compatible interface would use.

Run: python -m examples.hf_integration

The script does NOT replace HF's actual KV cache (that would require
custom CUDA kernels). It demonstrates the integration *interface* by:
  - Allocating blocks for each generated token
  - Forking for beam-search-like branching
  - Evicting older tokens
  - Verifying the rope's snapshot matches the actual cache length

This is the Phase 3 deliverable that runs on this machine. The full
vLLM integration is documented in src/rope/vllm_compat.py.
"""
import sys
import time
sys.path.insert(0, "src")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rope import RopeBlockManager, Tier


def main(model_name: str = "distilgpt2", prompt: str | None = None,
         max_new_tokens: int = 30, n_beams: int = 1):
    print(f"Loading {model_name} (CPU)...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    if prompt is None:
        prompt = "The quick brown fox jumps over the lazy dog."

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]
    print(f"\nPrompt: {prompt!r}  ({prompt_len} tokens)")

    # ----------------------------------------------------------------
    # Set up the rope-backed block manager as a "shadow" KV cache
    # ----------------------------------------------------------------
    num_kv_heads = getattr(model.config, "num_key_value_heads",
                           getattr(model.config, "num_attention_heads", 12))
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    bm = RopeBlockManager(
        block_size=16,
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    # ----------------------------------------------------------------
    # Run a single greedy generation, tracking the rope state
    # ----------------------------------------------------------------
    print(f"\nGenerating up to {max_new_tokens} tokens (greedy, beams={n_beams})...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=n_beams,
            do_sample=False,
        )
    elapsed = time.perf_counter() - t0
    new_tokens = out_ids.shape[1] - prompt_len
    print(f"  generated {new_tokens} tokens in {elapsed:.2f}s "
          f"({new_tokens / elapsed:.1f} tok/s)")

    # ----------------------------------------------------------------
    # Track the rope state alongside the generation
    # ----------------------------------------------------------------
    bm.create_sequence(1)
    # Allocate blocks for the prompt
    bm.allocate(1, n_tokens=prompt_len, kv_data=b"p" * prompt_len)
    # Each generated token is an append_slot
    for i in range(new_tokens):
        bm.append_slot(1, kv_data=bytes([i % 256]))

    # Verify the rope's snapshot matches the actual cache length
    snap = bm.get_snapshot(1)
    rope_length = sum(s[2] for s in snap)
    actual_length = prompt_len + new_tokens
    print(f"\n  rope snapshot length: {rope_length}")
    print(f"  actual cache length:  {actual_length}")
    assert rope_length == actual_length, "rope does not match actual cache length"

    # ----------------------------------------------------------------
    # Demonstrate branching (beam-search-like)
    # ----------------------------------------------------------------
    print("\nBranching demo (beam-search-like):")
    # Fork the main sequence into two
    bm.fork(parent_seq_id=1, child_seq_id=2)
    print(f"  forked seq 1 -> seq 2 (shared prefix of {prompt_len + new_tokens} tokens)")
    # Diverge: each side appends different data
    bm.allocate(1, n_tokens=10, kv_data=b"a" * 10)
    bm.allocate(2, n_tokens=10, kv_data=b"b" * 10)
    s1 = bm.get_snapshot(1)
    s2 = bm.get_snapshot(2)
    print(f"  seq 1 length: {sum(s[2] for s in s1)}")
    print(f"  seq 2 length: {sum(s[2] for s in s2)}")
    # Count leaves that share the same NodeID between both seqs
    # (i.e., neither seq wrote to them after fork).
    shared = sum(1 for a, b in zip(s1, s2) if a[0] == b[0])
    print(f"  shared {shared} leaf nodes by NodeID (out of {len(s1)})")

    # ----------------------------------------------------------------
    # Demonstrate eviction (WARM-tier spill)
    # ----------------------------------------------------------------
    print("\nEviction demo (WARM-tier spill):")
    bm.swap_out(1, n_tokens=prompt_len)
    rope = bm.rope
    leaves = list(rope._iter_leaves(bm.get_seq_root(1)))
    n_warm = sum(1 for l in leaves if l.tier == Tier.WARM)
    n_hot = sum(1 for l in leaves if l.tier == Tier.HOT)
    print(f"  after swap_out({prompt_len}): {n_warm} WARM leaves, {n_hot} HOT leaves")

    # ----------------------------------------------------------------
    # Final stats
    # ----------------------------------------------------------------
    print("\nFinal block-manager stats:")
    for k, v in bm.stats().items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    # ----------------------------------------------------------------
    # Output
    # ----------------------------------------------------------------
    generated_text = tokenizer.decode(out_ids[0][prompt_len:])
    print(f"\nGenerated text: {generated_text!r}")

    return bm, out_ids


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=30)
    p.add_argument("--num-beams", type=int, default=1)
    args = p.parse_args()
    main(args.model, args.prompt, args.max_new_tokens, args.num_beams)
