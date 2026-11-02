"""The cached forward pass must produce the same logits as the uncached one."""
import torch
import pytest

from model import DecoderOnlyTransformer

VOCAB, D_MODEL, HEADS, LAYERS, D_FF, MAX_LEN = 100, 64, 4, 2, 128, 64
ATOL = 1e-4


@pytest.fixture
def model():
    torch.manual_seed(0)
    return DecoderOnlyTransformer(
        num_tokens=VOCAB, d_model=D_MODEL, num_heads=HEADS,
        num_layers=LAYERS, d_ff=D_FF, max_len=MAX_LEN,
    ).eval()


def empty_caches(batch):
    shape = (batch, HEADS, MAX_LEN, D_MODEL // HEADS)
    return [(torch.zeros(shape), torch.zeros(shape)) for _ in range(LAYERS)]


def run_in_chunks(model, tokens, chunk_sizes):
    """Feed `tokens` through the cache in consecutive chunks; return all logits."""
    caches = empty_caches(tokens.size(0))
    pos, outs = 0, []
    for size in chunk_sizes:
        outs.append(model(tokens[:, pos:pos + size], kv_caches=caches, cache_seqlen=pos))
        pos += size
    assert pos == tokens.size(1)
    return torch.cat(outs, dim=1)


@pytest.mark.parametrize("chunks", [
    [1] * 12,          # decode only: one token at a time
    [8, 1, 1, 1, 1],   # prefill a prompt, then decode
    [5, 4, 1, 1, 1],   # chunked prefill: a multi-token chunk lands on a non-empty cache
])
@torch.no_grad()
def test_cached_matches_uncached(model, chunks):
    torch.manual_seed(1)
    tokens = torch.randint(0, VOCAB, (2, sum(chunks)))
    expected = model(tokens)
    actual = run_in_chunks(model, tokens, chunks)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=0)
