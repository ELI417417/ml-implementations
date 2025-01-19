"""
VGG19-based feature extractor for neural style transfer.

Returns intermediate activations from selected layers.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
    """Extract features from selected VGG19 layers for style transfer.

    Content layers: deeper layers capture semantic content.
    Style layers: multiple layers capture texture at different scales.
    """

    CONTENT_LAYERS = {"conv4_2"}   # single deep layer
    STYLE_LAYERS = {"conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"}

    def __init__(self, device: torch.device | None = None):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        # Freeze all parameters
        for param in self.features.parameters():
            param.requires_grad = False

        self.layer_map = self._build_layer_map()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def _build_layer_map(self) -> dict[str, int]:
        """Map VGG feature layer names to module indices."""
        layer_map = {}
        block, conv = 1, 1
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.Conv2d):
                name = f"conv{block}_{conv}"
                layer_map[name] = i
                conv += 1
            elif isinstance(layer, nn.ReLU):
                self.features[i] = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                block += 1
                conv = 1
        return layer_map

    def forward(self, x: torch.Tensor
                ) -> dict[str, torch.Tensor]:
        """Extract features for content and style layers."""
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            for name, idx in self.layer_map.items():
                if i == idx and (name in self.CONTENT_LAYERS
                                 or name in self.STYLE_LAYERS):
                    features[name] = x
        return features

    def content_layers(self) -> list[str]:
        return sorted(self.CONTENT_LAYERS)

    def style_layers(self) -> list[str]:
        return sorted(self.STYLE_LAYERS)
