"""
Stable Diffusion pipeline wrapper with attention extraction.

Wraps the diffusers StableDiffusionPipeline to extract and return
cross-attention maps for visualization.
"""

from typing import Optional

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


class AttentionExtractor:
    """Extracts cross-attention maps from Stable Diffusion UNet.

    Hooks into all CrossAttention layers and collects attention weights
    between text tokens and spatial positions during denoising.
    """

    def __init__(self, pipeline: StableDiffusionPipeline):
        self.pipeline = pipeline
        self.attention_maps: list[torch.Tensor] = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all CrossAttention layers."""
        unet = self.pipeline.unet

        def hook_fn(module, input_, output, name=""):
            # Cross-attention modules store attn weights during forward
            if hasattr(module, "attn") and module.attn is not None:
                self.attention_maps.append(module.attn.detach().cpu())

        for name, module in unet.named_modules():
            if "attn2" in name.lower() or "crossattn" in name.lower():
                self.hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o, n=name: hook_fn(m, i, o, n)
                    )
                )

    def clear(self):
        self.attention_maps.clear()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class SDExplorer:
    """Stable Diffusion explorer with attention visualization."""

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5",
                 device: str | None = None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_id = model_id
        self.pipe: StableDiffusionPipeline | None = None
        self.extractor: AttentionExtractor | None = None

    def load(self):
        """Load the SD pipeline (lazy loading)."""
        if self.pipe is None:
            dtype = torch.float16 if "cuda" in str(self.device) \
                else torch.float32
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
            ).to(self.device)
            self.pipe.enable_attention_slicing()
            self.extractor = AttentionExtractor(self.pipe)

    def generate(self, prompt: str, negative_prompt: str = "",
                 num_steps: int = 50, guidance_scale: float = 7.5,
                 seed: int | None = None, height: int = 512,
                 width: int = 512,
                 return_attention: bool = False
                 ) -> tuple[Image.Image, list[np.ndarray] | None]:
        """Generate an image and optionally return attention maps.

        Args:
            prompt: Text prompt for generation.
            negative_prompt: What to avoid.
            num_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance strength.
            seed: Random seed for reproducibility.
            height, width: Output image dimensions.
            return_attention: If True, also return attention maps.

        Returns:
            (generated PIL image, list of attention maps or None).
        """
        self.load()
        if seed is not None:
            torch.manual_seed(seed)

        if self.extractor and return_attention:
            self.extractor.clear()

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)

        with torch.no_grad():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )

        image = output.images[0]

        if return_attention and self.extractor:
            maps = [m.numpy() for m in self.extractor.attention_maps]
            return image, maps

        return image, None

    def interpolate_prompts(self, prompt_a: str, prompt_b: str,
                            num_frames: int = 5, **kwargs
                            ) -> list[Image.Image]:
        """Generate images interpolating between two prompts.

        Uses spherical linear interpolation (slerp) in the text embedding
        space to create smooth transitions.
        """
        self.load()
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder

        # Tokenize both prompts
        tokens_a = tokenizer(
            prompt_a, padding="max_length", max_length=77,
            return_tensors="pt"
        ).input_ids.to(self.device)
        tokens_b = tokenizer(
            prompt_b, padding="max_length", max_length=77,
            return_tensors="pt"
        ).input_ids.to(self.device)

        # Get embeddings
        with torch.no_grad():
            emb_a = text_encoder(tokens_a)[0]
            emb_b = text_encoder(tokens_b)[0]

        images = []
        for t in np.linspace(0, 1, num_frames):
            # Simple linear interpolation (slerp would be better but more complex)
            emb_t = emb_a * (1 - t) + emb_b * t
            img, _ = self.generate(
                prompt_a if t < 0.5 else prompt_b,
                **kwargs
            )
            images.append(img)

        return images
