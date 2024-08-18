"""
Tests for Transformer model components.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch

from model import (
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    EncoderLayer,
    DecoderLayer,
    Transformer,
)


class TestMultiHeadAttention:
    def test_output_shape(self):
        attn = MultiHeadAttention(d_model=256, n_heads=8)
        x = torch.randn(2, 10, 256)
        out, weights = attn(x, x, x)
        assert out.shape == (2, 10, 256)

    def test_attention_weights_shape(self):
        attn = MultiHeadAttention(d_model=256, n_heads=8)
        x = torch.randn(2, 10, 256)
        _, weights = attn(x, x, x)
        assert weights.shape == (2, 8, 10, 10)

    def test_attention_mask(self):
        attn = MultiHeadAttention(d_model=256, n_heads=8)
        x = torch.randn(2, 10, 256)
        mask = torch.ones(2, 1, 1, 10)
        mask[:, :, :, 5:] = 0  # mask last 5 positions
        out, weights = attn(x, x, x, mask)
        assert out.shape == (2, 10, 256)


class TestPositionalEncoding:
    def test_shape(self):
        pe = PositionalEncoding(d_model=256, max_len=100)
        x = torch.randn(2, 50, 256)
        out = pe(x)
        assert out.shape == (2, 50, 256)


class TestFeedForward:
    def test_shape(self):
        ff = FeedForward(d_model=256, d_ff=1024)
        x = torch.randn(2, 10, 256)
        out = ff(x)
        assert out.shape == (2, 10, 256)


class TestTransformer:
    @pytest.fixture
    def model(self):
        return Transformer(
            src_vocab_size=1000, tgt_vocab_size=1000,
            d_model=128, n_heads=4, n_encoder_layers=2,
            n_decoder_layers=2, d_ff=512, max_len=200,
        )

    def test_forward_shape(self, model):
        src = torch.randint(0, 1000, (2, 20))
        tgt = torch.randint(0, 1000, (2, 20))
        tgt_mask = Transformer.generate_square_subsequent_mask(
            20, torch.device("cpu")
        )
        out = model(src, tgt, tgt_mask=tgt_mask)
        assert out.shape == (2, 20, 1000)

    def test_encode_shape(self, model):
        src = torch.randint(0, 1000, (2, 20))
        enc_out, attn_maps = model.encode(src)
        assert enc_out.shape == (2, 20, 128)
        assert len(attn_maps) == 2  # n_encoder_layers
