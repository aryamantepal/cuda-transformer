"""
Benchmarks autoregressive inference:
  1. No KV-cache (baseline)
  2. KV-cache
  3. KV-cache + torch.compile

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
WARMUP_STEPS   = 5
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


def warmup(model, start_tokens, device, steps=WARMUP_STEPS):
    """Prime the model before timing — especially important after torch.compile."""
    with torch.no_grad():
        inp = start_tokens.clone()
        for _ in range(steps):
            logits, _ = model(inp)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            inp = torch.cat([inp, next_tok], dim=1)
    sync(device)


def bench_no_cache(model, start_tokens, device):
    model_input = start_tokens.clone()
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            logits, _ = model(model_input)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            model_input = torch.cat([model_input, next_tok], dim=1)
    sync(device)
    return time.perf_counter() - t0


def bench_with_cache(model, start_tokens, device):
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, kv_caches = model(start_tokens, kv_caches=None, offset=0)
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq_len = start_tokens.size(1)
        for step in range(MAX_NEW_TOKENS - 1):
            logits, kv_caches = model(next_tok, kv_caches=kv_caches, offset=seq_len + step)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(device)
    return time.perf_counter() - t0


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

    start_tokens = torch.randint(0, NUM_TOKENS, (1, PROMPT_LEN), device=device)

    # ── 1. No-cache baseline ─────────────────────────────────────────────────
    model = build_model(device)
    warmup(model, start_tokens, device)
    elapsed_no_cache = bench_no_cache(model, start_tokens, device)

    print("=" * 50)
    print_results("NO KV-CACHE (baseline)", elapsed_no_cache)

    # ── 2. KV-cache ──────────────────────────────────────────────────────────
    model = build_model(device)
    warmup(model, start_tokens, device)
    elapsed_kv = bench_with_cache(model, start_tokens, device)

    print("=" * 50)
    print_results("KV-CACHE", elapsed_kv)

    # ── 3. KV-cache + torch.compile ──────────────────────────────────────────
    # Compile only the individual layers, not the full generate loop.
    # The generate loop manages growing kv_cache tensors (new shape each step)
    # which would cause inductor to recompile on every iteration if we compiled
    # the whole model. Compiling per-layer keeps tensor shapes stable inside
    # each compiled unit (single token in, fixed d_model out).
    model = build_model(device)
    backend = "aot_eager" if device.type == "mps" else "inductor"
    for i, layer in enumerate(model.layers):
        model.layers[i] = torch.compile(layer, dynamic=True, backend=backend)

    print("=" * 50)
    print(f"  Compiling... backend={backend} (first warmup will be slow — that's normal)")
    warmup(model, start_tokens, device)
    elapsed_compiled = bench_with_cache(model, start_tokens, device)

    print_results("KV-CACHE + torch.compile", elapsed_compiled)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("=" * 50)
    print(f"  KV-cache speedup          : {elapsed_no_cache / elapsed_kv:.2f}x")
    print(f"  compile speedup (vs cache): {elapsed_kv / elapsed_compiled:.2f}x")
    print(f"  compile speedup (vs base) : {elapsed_no_cache / elapsed_compiled:.2f}x")
    print("=" * 50)


if __name__ == "__main__":
    main()
