"""
Transformer implementation from "Attention Is All You Need" (Vaswani et al., 2017).

Multi-head scaled dot-product attention, positional encoding,
encoder-decoder architecture — all built from scratch in PyTorch.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention."""

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        # Linear projections and split into heads
        Q = self.W_q(query).view(batch_size, -1, self.n_heads,
                                 self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads,
                                 self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.W_o(context)
        return output, attn_weights


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int = 512, max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int = 512, d_ff: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder layer: self-attention + feed-forward."""

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x, attn_weights


class DecoderLayer(nn.Module):
    """Single decoder layer: self-attention + cross-attention + feed-forward."""

    def __init__(self, d_model: int = 512, n_heads: int = 8,
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor,
                src_mask: torch.Tensor | None = None,
                tgt_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self_attn_out, self_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_out, cross_weights = self.cross_attn(x, enc_out, enc_out,
                                                    src_mask)
        x = self.norm2(x + self.dropout(cross_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x, self_weights, cross_weights


class Transformer(nn.Module):
    """Full Transformer model — encoder + decoder."""

    def __init__(self, src_vocab_size: int = 30000,
                 tgt_vocab_size: int = 30000,
                 d_model: int = 512, n_heads: int = 8,
                 n_encoder_layers: int = 6, n_decoder_layers: int = 6,
                 d_ff: int = 2048, max_len: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder_embed = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def encode(self, src: torch.Tensor,
               src_mask: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.encoder_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        attn_maps = []
        for layer in self.encoder_layers:
            x, attn = layer(x, src_mask)
            attn_maps.append(attn)
        return x, attn_maps

    def decode(self, tgt: torch.Tensor, enc_out: torch.Tensor,
               src_mask: torch.Tensor | None = None,
               tgt_mask: torch.Tensor | None = None
               ) -> tuple[torch.Tensor, list[torch.Tensor],
                          list[torch.Tensor]]:
        x = self.decoder_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        self_attns, cross_attns = [], []
        for layer in self.decoder_layers:
            x, self_a, cross_a = layer(x, enc_out, src_mask, tgt_mask)
            self_attns.append(self_a)
            cross_attns.append(cross_a)
        return x, self_attns, cross_attns

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor | None = None,
                tgt_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        enc_out, _ = self.encode(src, src_mask)
        dec_out, _, _ = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.output_proj(dec_out)

    @staticmethod
    def generate_square_subsequent_mask(sz: int,
                                        device: torch.device) -> torch.Tensor:
        """Causal mask for autoregressive decoding."""
        return torch.triu(torch.ones(sz, sz, device=device)
                          * float("-inf"), diagonal=1)
