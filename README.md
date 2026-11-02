# cuda-transformer

A decoder-only transformer built from scratch to understand and benchmark the inference optimizations that actually matter in practice. The focus is on *why* each technique works, not just that it does — so each experiment isolates one variable and measures the result on real hardware.

**Scope:** the model is untrained. `train.py` feeds it random tokens and random labels so training throughput and memory can be profiled, so its outputs are meaningless text. The benchmarks measure speed only (tokens/sec, ms/token), never output quality.

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

**Result** (`results/t4_current.txt`: Tesla T4, torch 2.11.0, fp32, batch 32, 256-token prompt, 200 new tokens):

| | ms per step | Tok/s per sequence | Tok/s total (x32) |
|---|---|---|---|
| No cache | 230.58 | 4.3 | 138 |
| Static KV-cache | 5.68 | 175.9 | 5,629 |
| **Speedup** | | **40.6x** | |

Each step makes one token for each of the 32 sequences. The no-cache baseline is weaker than it needs to be: it computes output scores for every position and keeps only the last one, so part of the 40.6x comes from that waste.

### 2. torch.compile

**What:** `torch.compile` turns PyTorch code into fused GPU kernels via TorchInductor.

**What the logs show** (`results/t4_compile_before.txt` vs `results/t4_compile_after.txt`, same settings as above, run with `TORCH_LOGS=recompiles`):

| | Before: growing `torch.cat` cache, per-layer compile | After: static buffer, whole-model compile |
|---|---|---|
| Cache, no compile | 118.1 tok/s/seq | 175.4 tok/s/seq |
| Cache + compile | 35.7 tok/s/seq (0.30x) | 176.9 tok/s/seq (1.01x) |
| Recompiles | 2 | 1 |

- **The old code did not recompile on every step.** It recompiled twice (once when switching from prompt to decode, once on a memory-layout change), then `dynamic=True` handled the growing length.
- **The change still fixed the slowdown:** compile went from 0.30x to about 1.0x. Several things changed at once (static buffer, whole-model compile, new attention code), so these runs can't say which one did it.
- **Dropping `torch.cat` sped up the uncompiled path 1.48x.** `torch.cat` copies the whole cache every step; the static buffer writes in place.
- **Compile still doesn't beat no compile** (0.98x and 1.01x across two runs). A decode step reads about 490 MB, which the T4 could do in about 1.5 ms, but a step takes 5.7 ms. Most of the time likely goes to launching hundreds of small GPU kernels, which compile's default mode doesn't remove. CUDA graphs (`mode="reduce-overhead"`) could, but they need the same tensor shapes every step.

### 3. FP16

**What:** Cast weights, activations and the KV cache to float16. Token indices stay int64.

**Result** (`results/t4_current.txt`): 3.29 ms per step vs 5.68 ms in fp32, a **1.73x** speedup (303.5 tok/s/seq, 9,712 total). Decode is limited by how fast it can read weights and cache from memory, not by math, so halving the bytes caps the gain near 2x. The T4's much larger fp16 math advantage barely matters here.

## Results

Raw benchmark outputs are in `results/`. The README quotes only the current logs.

**Current code** (Tesla T4, torch 2.11.0, batch 32, 256-token prompt, 200 new tokens):

| File | Notes |
|---|---|
| `t4_current.txt` | All variants, plus run settings at the end |
| `t4_compile_after.txt` | Same, with `TORCH_LOGS=recompiles` |
| `t4_compile_before.txt` | Old code (commit `b13d144`), with `TORCH_LOGS=recompiles` |

**Historical** (March 2026 code, before the static buffer and the prefill mask fix; most settings not recorded in the logs):

| File | Hardware |
|---|---|
| `baseline.txt`, `flash_attn.txt`, `kv_cache_plus_compile.txt` | M2 Mac (MPS) |
| `t4_baseline.txt`, `t4_longer_seq.txt`, `t4_long_prompt.txt`, `t4_fp16.txt`, `t4_fp16_batch32.txt` | T4 GPU |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tests: cached inference must match the uncached forward pass
python -m pytest tests

# Training (synthetic data, profiling)
python train.py

# Inference benchmark
python inference.py
```

Automatically picks the best available device: CUDA → MPS → CPU.
