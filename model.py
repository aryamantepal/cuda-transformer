import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.optim import Adam


class PositionEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        embedding_index = torch.arange(0, d_model, 2).float()
        div_term = 1 / torch.tensor(10000.0) ** (embedding_index / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, word_embeddings, offset=0):
        T = word_embeddings.size(1)
        return word_embeddings + self.pe[offset:offset + T, :].unsqueeze(0).to(word_embeddings.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None, cache_seqlen=0):
        """
        Training / prefill (kv_cache=None):
            - x: (B, T, C), full sequence
            - causal mask applied, no cache read/write

        Static-cache decode (kv_cache=(k_buf, v_buf)):
            - x: (B, 1, C), single new token
            - writes k/v into pre-allocated buffers at position cache_seqlen
            - attends over [:cache_seqlen+1] — buffer shape is always max_len, never reallocated
        """
        B, T, C = x.shape

        q     = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_buf, v_buf = kv_cache
            # In-place write — buffer shape (B, H, max_len, head_dim) never changes.
            k_buf[:, :, cache_seqlen:cache_seqlen + T, :] = k_new
            v_buf[:, :, cache_seqlen:cache_seqlen + T, :] = v_new
            # Slice only the filled portion for attention.
            k = k_buf[:, :, :cache_seqlen + T, :]
            v = v_buf[:, :, :cache_seqlen + T, :]
            # Single query attends to all past tokens — is_causal=False is correct here.
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k_new, v_new, is_causal=True)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(attn_out)


class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, kv_cache=None, cache_seqlen=0):
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache, cache_seqlen=cache_seqlen)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderOnlyTransformer(L.LightningModule):
    def __init__(self, num_tokens=32000, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=512):
        super().__init__()
        self.we = nn.Embedding(num_tokens, d_model)
        self.pe = PositionEncoding(d_model=d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc_layer = nn.Linear(d_model, num_tokens)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, token_ids, kv_caches=None, cache_seqlen=0):
        """
        Args:
            token_ids:    (B, T) integer token ids
            kv_caches:    list of (k_buf, v_buf) pre-allocated tensors per layer, or None
            cache_seqlen: number of tokens already written into the cache buffers
        Returns:
            logits: (B, T, vocab)
        """
        x = self.pe(self.we(token_ids), offset=cache_seqlen)

        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, kv_cache=cache, cache_seqlen=cache_seqlen)

        return self.fc_layer(self.norm(x))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=3e-4)

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        logits = self.forward(input_tokens)
        loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss
