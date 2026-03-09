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
        # Bug fix: use `offset` so cached inference gets the right positional encoding.
        # During prefill: offset=0, T=full prompt length -> pe[0:T]
        # During decode:  offset=current_seq_len, T=1 -> pe[offset:offset+1]
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

    def forward(self, x, mask=None, kv_cache=None):
        B, T, C = x.shape

        q     = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v_new = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k_new], dim=2)
            v = torch.cat([v_prev, v_new], dim=2)
        else:
            k, v = k_new, v_new

        # (B, H, T_q, T_k)
        sims = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            sims = sims.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)

        attn_probs = F.softmax(sims, dim=-1)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        updated_cache = (k, v)
        return self.W_o(attn_out), updated_cache


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

    def forward(self, x, mask=None, kv_cache=None):
        attn_out, updated_cache = self.attn(self.norm1(x), mask=mask, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, updated_cache


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

    def forward(self, token_ids, kv_caches=None, offset=0):
        """
        Args:
            token_ids:  (B, T) integer token ids
            kv_caches:  list of (k, v) tensors per layer, or None for training/prefill
            offset:     how many tokens have already been processed (for positional encoding)
        Returns:
            logits:     (B, T, vocab)
            new_caches: list of updated (k, v) per layer
        """
        B, T = token_ids.shape
        x = self.we(token_ids)
        x = self.pe(x, offset=offset)

        # Causal mask: only over the current input tokens (T_q x T_q for prefill, 1x1 for decode)
        mask = ~torch.tril(torch.ones(T, T, device=token_ids.device)).bool()

        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, updated_cache = layer(x, mask=mask, kv_cache=cache)
            new_caches.append(updated_cache)

        x = self.norm(x)
        logits = self.fc_layer(x)
        return logits, new_caches

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=3e-4)

    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        logits, _ = self.forward(input_tokens)
        loss = self.loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss
