"""
Benchmarks autoregressive inference:
  1. No KV-cache (baseline)
  2. Static KV-cache (pre-allocated buffers, in-place writes)
  3. Static KV-cache + torch.compile
  4. Static KV-cache + FP16

torch.compile notes (see results/t4_compile_*.txt):
  The old code grew the cache with torch.cat each step and compiled each layer
  separately.  It recompiled only twice, yet compiled decode ran at 0.30x.  With
  pre-allocated buffers written in place and the whole forward compiled, it runs
  at about 1.0x: decode is likely dominated by kernel-launch overhead, which
  default-mode compile doesn't remove.

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

PROMPT_LEN     = 256
MAX_NEW_TOKENS = 200
WARMUP_STEPS   = 3
# ─────────────────────────────────────────────────────────────────────────────


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_model(device, fp16=False):
    model = DecoderOnlyTransformer(
        num_tokens=NUM_TOKENS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_len=MAX_LEN,
    ).to(device)
    if fp16:
        model = model.half()
    model.eval()
    return model


def alloc_kv_caches(batch_size, max_len, device, dtype=torch.float32):
    """Pre-allocate fixed-size KV cache buffers for all layers."""
    head_dim = D_MODEL // NUM_HEADS
    caches = []
    for _ in range(NUM_LAYERS):
        k = torch.zeros(batch_size, NUM_HEADS, max_len, head_dim, device=device, dtype=dtype)
        v = torch.zeros_like(k)
        caches.append((k, v))
    return caches


def generate_no_cache(model, start_tokens, device):
    model_input = start_tokens.clone()
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits = model(model_input)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            model_input = torch.cat([model_input, next_tok], dim=1)
    sync(device)
    return time.perf_counter() - t0


def generate_static_cache(model, start_tokens, device, dtype=torch.float32):
    """
    Prefill the prompt in one shot, then decode one token at a time using
    pre-allocated static KV cache buffers.  No tensor allocation inside the loop.
    """
    B = start_tokens.size(0)
    kv_caches = alloc_kv_caches(B, MAX_LEN, device, dtype=dtype)

    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        # Prefill: process the full prompt, filling cache positions [0, PROMPT_LEN)
        logits = model(start_tokens, kv_caches=kv_caches, cache_seqlen=0)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Decode: one token at a time, cache_seqlen advances each step
        for step in range(MAX_NEW_TOKENS - 1):
            logits = model(next_tok, kv_caches=kv_caches, cache_seqlen=PROMPT_LEN + step)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(device)
    return time.perf_counter() - t0


def warmup_model(model, start_tokens, device, steps=WARMUP_STEPS, use_cache=False, dtype=torch.float32):
    with torch.no_grad():
        if use_cache:
            B = start_tokens.size(0)
            kv = alloc_kv_caches(B, MAX_LEN, device, dtype=dtype)
            logits = model(start_tokens, kv_caches=kv, cache_seqlen=0)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for s in range(steps - 1):
                logits = model(next_tok, kv_caches=kv, cache_seqlen=PROMPT_LEN + s)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        else:
            inp = start_tokens.clone()
            for _ in range(steps):
                logits = model(inp)
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                inp = torch.cat([inp, next_tok], dim=1)
    sync(device)


def print_results(label, elapsed):
    tps = MAX_NEW_TOKENS / elapsed
    mpt = elapsed / MAX_NEW_TOKENS * 1000
    print(f"  {label}")
    print(f"  Tokens/sec : {tps:.1f}")
    print(f"  ms/token   : {mpt:.2f}")
    print(f"  Total      : {elapsed:.3f}s")


def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"torch version: {torch.__version__}\n")

    batch_size = 4 if device.type != "cuda" else 32
    start_tokens = torch.randint(0, NUM_TOKENS, (batch_size, PROMPT_LEN), device=device)

    # ── 1. No-cache baseline ──────────────────────────────────────────────────
    model = build_model(device)
    warmup_model(model, start_tokens, device, use_cache=False)
    elapsed_no_cache = generate_no_cache(model, start_tokens, device)

    print("=" * 50)
    print_results("NO KV-CACHE (baseline)", elapsed_no_cache)

    # ── 2. Static KV-cache ────────────────────────────────────────────────────
    model = build_model(device)
    warmup_model(model, start_tokens, device, use_cache=True)
    elapsed_kv = generate_static_cache(model, start_tokens, device)

    print("=" * 50)
    print_results("STATIC KV-CACHE", elapsed_kv)

    # ── 3. Static KV-cache + torch.compile ───────────────────────────────────
    # Compile model.forward (not individual layers).  The static buffer means
    # tensor shapes inside the graph are fixed; only cache_seqlen (a scalar)
    # changes each step.  dynamic=True treats it as symbolic, so decode steps
    # share one compiled graph (prefill gets its own).
    model = build_model(device)
    backend = "aot_eager" if device.type == "mps" else "inductor"
    model.forward = torch.compile(model.forward, dynamic=True, backend=backend)

    print("=" * 50)
    print(f"  Compiling model.forward... backend={backend}")
    print("  (first warmup will be slow — inductor traces on first call)")
    warmup_model(model, start_tokens, device, use_cache=True)
    elapsed_compiled = generate_static_cache(model, start_tokens, device)

    print_results("STATIC KV-CACHE + torch.compile", elapsed_compiled)

    # ── 4. Static KV-cache + FP16 ────────────────────────────────────────────
    # FP16 halves the bytes read per decode step.  Decode is memory-bound, so
    # the gain is capped near 2x (1.73x measured on T4).
    if device.type in ("cuda", "mps"):
        model = build_model(device, fp16=True)
        warmup_model(model, start_tokens, device, use_cache=True, dtype=torch.float16)
        elapsed_fp16 = generate_static_cache(model, start_tokens, device, dtype=torch.float16)

        print("=" * 50)
        print_results("STATIC KV-CACHE + FP16", elapsed_fp16)
    else:
        elapsed_fp16 = None

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"  KV-cache speedup             : {elapsed_no_cache / elapsed_kv:.2f}x")
    print(f"  compile speedup (vs cache)   : {elapsed_kv / elapsed_compiled:.2f}x")
    print(f"  compile speedup (vs base)    : {elapsed_no_cache / elapsed_compiled:.2f}x")
    if elapsed_fp16 is not None:
        print(f"  FP16 speedup (vs cache)      : {elapsed_kv / elapsed_fp16:.2f}x")
        print(f"  FP16 speedup (vs base)       : {elapsed_no_cache / elapsed_fp16:.2f}x")
    print("=" * 50)

    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("CUDA Memory Summary (post-benchmark)")
        print("=" * 60)
        print(torch.cuda.memory_summary())


if __name__ == "__main__":
    main()
