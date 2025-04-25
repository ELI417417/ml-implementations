"""
Tests for LoRA layer implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import torch
import torch.nn as nn

from lora import LoRALinear, LoRAConfig, inject_lora, count_trainable_params


class TestLoRALinear:
    @pytest.fixture
    def config(self):
        return LoRAConfig(rank=4, alpha=8, dropout=0.0)

    @pytest.fixture
    def base_linear(self):
        return nn.Linear(16, 32)

    def test_forward_shape(self, config, base_linear):
        lora = LoRALinear.from_linear(base_linear, config)
        x = torch.randn(2, 16)
        out = lora(x)
        assert out.shape == (2, 32)

    def test_base_frozen(self, config, base_linear):
        lora = LoRALinear.from_linear(base_linear, config)
        assert not lora.base.weight.requires_grad

    def test_lora_trainable(self, config, base_linear):
        lora = LoRALinear.from_linear(base_linear, config)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_scaling(self, config, base_linear):
        lora = LoRALinear.from_linear(base_linear, config)
        assert lora.scaling == 2.0  # alpha/rank = 8/4

    def test_merge(self, config, base_linear):
        lora = LoRALinear.from_linear(base_linear, config)
        merged = lora.merge()
        assert isinstance(merged, nn.Linear)
        # base_linear is (16, 32), merged weight is (out_features, in_features) = (32, 16)
        assert merged.weight.shape == (32, 16)

    def test_merge_equivalence(self, config):
        base = nn.Linear(8, 8)
        lora = LoRALinear.from_linear(base, config)
        # Set A and B to specific values
        torch.nn.init.ones_(lora.lora_A)
        torch.nn.init.ones_(lora.lora_B)
        x = torch.randn(3, 8)
        out_lora = lora(x)
        merged = lora.merge()
        out_merged = merged(x)
        torch.testing.assert_close(out_lora, out_merged)


class TestCountParams:
    def test_all_trainable(self):
        model = nn.Linear(10, 5)
        trainable, total = count_trainable_params(model)
        assert trainable == total == 55  # 50 weights + 5 bias

    def test_some_frozen(self):
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))
        for p in model[0].parameters():
            p.requires_grad = False
        trainable, total = count_trainable_params(model)
        assert total == 67  # 55 + 12
        assert trainable == 12  # only second layer
