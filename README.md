# cuda-transformer

A decoder-only transformer built from scratch to understand and benchmark the inference optimizations that actually matter in practice. The focus is on *why* each technique works, not just that it does — so each experiment isolates one variable and measures the result on real hardware.

## Architecture

Standard GPT-style decoder-only transformer, implemented in PyTorch + Lightning:

- 51.7M parameters
- 6 decoder blocks, 8 heads, d_model=512, d_ff=2048
- Sinusoidal positional encoding
- Pre-norm (LayerNorm before attention and FFN)
- Causal self-attention with `scaled_dot_product_attention`

## Experiments

All inference benchmarks are in `inference.py`. Training is in `train.py` (synthetic data, used for profiling training throughput and memory).

### 1. KV-Cache

**What:** During autoregressive decode, each new token needs to attend to all previous tokens. Without a cache, you recompute keys and values for the entire sequence every step — O(n²) total work. A KV-cache stores those tensors and reuses them, making each decode step O(n) instead.

**Result (T4 GPU, seq=256+200 tokens, batch=32):**

| | Tokens/sec | ms/token |
|---|---|---|
| No cache | 161 | 6.19 |
| KV-cache | 296 | 3.37 |
| **Speedup** | **1.84x** | |

The speedup grows with sequence length — at short sequences the overhead of managing the cache erodes the gain.

### 2. torch.compile

**What:** `torch.compile` lowers PyTorch operations into optimized kernels via TorchInductor. For inference, the promise is fused kernels, reduced Python overhead, and potentially CUDA graph capture.

**The problem with the naive approach:** The original implementation grew the KV cache with `torch.cat` each decode step, producing a tensor of a new shape `(B, H, T+1, head_dim)` on every token. Even with `dynamic=True`, inductor recompiles when output shapes change — so it was recompiling on every single decode step. This made "compiled" inference 6-100x *slower* than the baseline in initial results.

**The fix — static KV cache buffers:** Pre-allocate fixed-size buffers `(B, H, max_len, head_dim)` before the decode loop. Each step, write new k/v in-place at position `cache_seqlen`, then slice `[:, :, :cache_seqlen+T]` for attention. The buffer tensor shape never changes; only a scalar endpoint changes. Inductor handles that with `dynamic=True` without recompiling.

**Also changed:** Compile `model.forward` directly rather than individual layers. Per-layer compilation was a workaround for the shape-change problem that didn't actually solve it.

### 3. FP16

**What:** Cast all weights and activations to float16. On CUDA hardware with tensor cores (T4, A100, etc.), FP16 matrix multiplications run at ~8x the throughput of FP32. Memory bandwidth is halved, which matters a lot for memory-bound decode steps.

**Note:** Token indices stay int64 — only the float tensors (weights, activations, KV cache) move to FP16. The embedding layer handles the int→float transition automatically.

## Results

Raw benchmark outputs are saved in `results/`:

| File | Hardware | Notes |
|---|---|---|
| `baseline.txt` | M2 Mac (MPS) | Initial KV-cache experiment |
| `flash_attn.txt` | M2 Mac (MPS) | After refactoring attention |
| `kv_cache_plus_compile.txt` | M2 Mac (MPS) | Naive compile — shows regression |
| `t4_baseline.txt` | T4 GPU | Short sequence (50 new tokens) |
| `t4_longer_seq.txt` | T4 GPU | Longer sequence (200 new tokens, prompt=256) |

## Running

```bash
# Training (synthetic data, profiling)
python train.py

# Inference benchmark
python inference.py
```

Automatically picks the best available device: CUDA → MPS → CPU.
