"""
LoRA (Low-Rank Adaptation) implementation from scratch.

Based on Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.

LoRALinear stores two low-rank matrices:
  - A: (in_features, rank) — initialization: Kaiming uniform
  - B: (rank, out_features) — initialization: zeros
  Forward: h = Wx + (alpha/rank) * (x @ A) @ B
"""

import math
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRATarget(str, Enum):
    Q = "q_proj"
    K = "k_proj"
    V = "v_proj"
    O = "o_proj"
    ALL = "all"


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] | None = None
    bias: str = "none"


def default_target_modules() -> list[str]:
    return [LoRATarget.Q.value, LoRATarget.V.value]


class LoRALinear(nn.Module):
    """Linear layer with LoRA low-rank decomposition.

    A has shape (in_features, rank).  B has shape (rank, out_features).
    LoRA: output = base(x) + scaling * (x @ A) @ B
          scaling = alpha / rank
    """

    def __init__(self, in_features: int, out_features: int,
                 config: LoRAConfig, base_layer: nn.Linear | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = config.rank
        self.alpha = config.alpha
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        if base_layer is not None:
            self.base = base_layer
            for p in self.base.parameters():
                p.requires_grad = False
        else:
            self.base = nn.Linear(in_features, out_features, bias=False)

        # A: (in_features, rank) — maps input to low-rank space
        self.lora_A = nn.Parameter(torch.zeros(in_features, config.rank))
        # B: (rank, out_features) — maps low-rank to output
        self.lora_B = nn.Parameter(torch.zeros(config.rank, out_features))
        self.reset_lora_parameters()

        if config.bias == "all" or config.bias == "lora_only":
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        # x: (batch, in_features), A: (in_features, rank)
        # lora_out: (batch, rank) then @ B (rank, out_features) -> (batch, out_features)
        lora_out = F.linear(x, self.lora_A.T)
        if self.dropout is not None:
            lora_out = self.dropout(lora_out)
        lora_out = F.linear(lora_out, self.lora_B.T)
        result = result + self.scaling * lora_out
        if self.bias is not None:
            result = result + self.bias
        return result

    def merge(self) -> nn.Linear:
        """Merge LoRA into base: W_merged = W_base + (alpha/rank) * (A @ B)"""
        # A: (in_features, rank), B: (rank, out_features)
        # A @ B: (in_features, out_features) — transpose to match Linear weight shape (out_features, in_features)
        delta = self.scaling * (self.lora_A.data @ self.lora_B.data).T
        merged = nn.Linear(self.in_features, self.out_features,
                          bias=self.base.bias is not None)
        merged.weight.data = self.base.weight.data + delta
        if self.base.bias is not None:
            merged.bias.data = self.base.bias.data
        return merged

    @classmethod
    def from_linear(cls, linear: nn.Linear, config: LoRAConfig) -> "LoRALinear":
        return cls(linear.in_features, linear.out_features, config,
                   base_layer=linear)


def inject_lora(model: nn.Module, config: LoRAConfig,
                target_modules: list[str] | None = None) -> nn.Module:
    if target_modules is None:
        target_modules = default_target_modules()

    replaced = 0
    for name, module in model.named_modules():
        if not any(t in name for t in target_modules):
            continue
        if not isinstance(module, nn.Linear):
            continue

        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        attr_name = name.rsplit(".", 1)[-1]
        parent = model.get_submodule(parent_name) if parent_name else model

        lora_layer = LoRALinear.from_linear(module, config)
        setattr(parent, attr_name, lora_layer)
        replaced += 1

    print(f"Injected LoRA into {replaced} layers (targets: {target_modules})")
    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
