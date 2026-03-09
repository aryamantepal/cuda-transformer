"""
Benchmarks autoregressive inference with and without KV-caching.

Usage:
    python inference.py
"""

import time
import torch

from model import DecoderOnlyTransformer

# ── Config ────────────────────────────────────────────────────────────────────
NUM_TOKENS     = 32000
D_MODEL        = 512
NUM_HEADS      = 8
NUM_LAYERS     = 6
D_FF           = 2048
MAX_LEN        = 512

PROMPT_LEN     = 10
MAX_NEW_TOKENS = 50
WARMUP_STEPS   = 5          # short warmup to prime CUDA kernels
# ─────────────────────────────────────────────────────────────────────────────


def build_model(device):
    model = DecoderOnlyTransformer(
        num_tokens=NUM_TOKENS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
    ).to(device)
    model.eval()
    return model


def warmup(model, start_tokens, steps=WARMUP_STEPS):
    """Run a few no-grad forward passes to prime CUDA kernels before timing."""
    with torch.no_grad():
        inp = start_tokens.clone()
        for _ in range(steps):
            logits, _ = model(inp)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            inp = torch.cat([inp, next_tok], dim=1)


def bench_no_cache(model, start_tokens, max_new_tokens, device):
    """Baseline: re-run the full growing sequence every step. O(T^2) cost."""
    model_input = start_tokens.clone()

    sync(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(model_input)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            model_input = torch.cat([model_input, next_tok], dim=1)

    sync(device)
    elapsed = time.perf_counter() - t0
    return elapsed


def bench_with_cache(model, start_tokens, max_new_tokens, device):
    """
    KV-cache: process the prompt once (prefill), then pass one token per step.
    O(T) amortized cost — each decode step is constant time.
    """
    sync(device)
    t0 = time.perf_counter()

    with torch.no_grad():
        # ── Prefill ──────────────────────────────────────────────────────────
        # Process the entire prompt, populate the KV cache.
        logits, kv_caches = model(start_tokens, kv_caches=None, offset=0)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq_len = start_tokens.size(1)

        # ── Decode ───────────────────────────────────────────────────────────
        # Pass ONE new token at a time; offset tells PE which position this is.
        for step in range(max_new_tokens - 1):
            logits, kv_caches = model(
                next_tok,
                kv_caches=kv_caches,
                offset=seq_len + step,   # fix: correct absolute position
            )
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    sync(device)
    elapsed = time.perf_counter() - t0
    return elapsed


def print_results(label, elapsed, n_tokens):
    tps = n_tokens / elapsed
    mpt = elapsed / n_tokens * 1000
    print(f"  Generated {n_tokens} tokens in {elapsed:.3f}s")
    print(f"  Tokens/sec   : {tps:.1f}")
    print(f"  ms/token     : {mpt:.2f}")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    """Device-agnostic synchronization before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    device = get_device()
    print(f"Device: {device}\n")

    model = build_model(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    start_tokens = torch.randint(0, NUM_TOKENS, (1, PROMPT_LEN), device=device)

    # Warmup before any timing
    print("Warming up...")
    warmup(model, start_tokens)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── No-cache baseline ────────────────────────────────────────────────────
    print("=" * 60)
    print("WITHOUT KV-CACHE (baseline)")
    print("=" * 60)
    elapsed_baseline = bench_no_cache(model, start_tokens, MAX_NEW_TOKENS, device)
    print_results("baseline", elapsed_baseline, MAX_NEW_TOKENS)
    if torch.cuda.is_available():
        peak_baseline = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU mem : {peak_baseline:.1f} MiB")
        torch.cuda.reset_peak_memory_stats()

    # ── KV-cache ─────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("WITH KV-CACHE")
    print("=" * 60)
    elapsed_kv = bench_with_cache(model, start_tokens, MAX_NEW_TOKENS, device)
    print_results("kv-cache", elapsed_kv, MAX_NEW_TOKENS)
    if torch.cuda.is_available():
        peak_kv = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU mem : {peak_kv:.1f} MiB")

    # ── Summary ──────────────────────────────────────────────────────────────
    speedup = elapsed_baseline / elapsed_kv
    print()
    print("=" * 60)
    print(f"  Speedup: {speedup:.2f}x  ({elapsed_baseline:.3f}s → {elapsed_kv:.3f}s)")
    print("=" * 60)

    if torch.cuda.is_available():
        print("\nFull CUDA memory summary:")
        print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
