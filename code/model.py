# Purpose: loads/freezes/saves all of the model components
import torch
import torch.nn as nn
from diffusers import (
    StableDiffusionPipeline,  # wraps all components for easy inference
    UNet2DConditionModel,  # the noise predictor - the only thing being fine-tuned
    AutoencoderKL,  # VAE: compresses images to/from latent space
    DDPMScheduler  # controls the noise schedule during training
)
from transformers import CLIPTextModel, CLIPTokenizer

# CLIP is the text encoder. It converts your text prompt into embeddings (encoder_hidden_states) that guide that UNet's denoising.
# CLIPTokenizer turns raw text into token IDs first.
# runwayml/stable-diffusion-v1-5/
#   ├── tokenizer/
#   │   ├── vocab.json
#   │   └── merges.txt
#   ├── text_encoder/
#   │   └── pytorch_model.bin
#   ├── vae/
#   │   └── diffusion_pytorch_model.bin
#   ├── unet/
#   │   └── diffusion_pytorch_model.bin
#   └── scheduler/
#       └── scheduler_config.json
MODEL_ID = "runwayml/stable-diffusion-v1-5"


def freeze_component(component):
    """
    One clean reusable function to freeze any component.
    """
    component.requires_grad_(False)


def save_unet(unet, output_dir: str):
    """
    Save only the UNet weights.
    """
    torch.save(unet.state_dict(), f"{output_dir}/unet.pt")


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with two low-rank trainable matrices A and B.
    The effective weight update is B(A(x)) * scale, starting from zero (B init to 0).
    """
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: int = 4):
        super().__init__()
        self.linear = linear
        device = linear.weight.device
        # lora params stay float32 so GradScaler can unscale them
        self.lora_A = nn.Linear(linear.in_features, rank, bias=False).to(device=device)
        self.lora_B = nn.Linear(rank, linear.out_features, bias=False).to(device=device)
        self.scale = alpha / rank
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # cast x to float32 for LoRA, then cast result back to match x's dtype
        lora_out = self.lora_B(self.lora_A(x.float())).to(x.dtype)
        return self.linear(x) + lora_out * self.scale


DEFAULT_TARGET_MODULES = {"to_q", "to_k", "to_v", "to_out"}


def inject_lora(unet, rank: int = 4, alpha: int = 4, target_modules: set = DEFAULT_TARGET_MODULES):
    """
    Replace attention projection linears in every attention block with LoRALinear wrappers.
    target_modules controls which projections are adapted — any subset of:
        {"to_q", "to_k", "to_v", "to_out"}
    e.g. target_modules={"to_q", "to_v"} is a common lightweight choice.
    """
    for module in unet.modules():
        if hasattr(module, 'to_q'):
            if "to_q"   in target_modules:
                module.to_q      = LoRALinear(module.to_q,      rank, alpha)
            if "to_k"   in target_modules:
                module.to_k      = LoRALinear(module.to_k,      rank, alpha)
            if "to_v"   in target_modules:
                module.to_v      = LoRALinear(module.to_v,      rank, alpha)
            if "to_out" in target_modules:
                module.to_out[0] = LoRALinear(module.to_out[0], rank, alpha)


def lora_parameters(unet):
    """Return only the LoRA trainable parameters (lora_A and lora_B weights)."""
    return [p for name, p in unet.named_parameters() if "lora_" in name]


def save_lora(unet, output_dir: str):
    """Save only the LoRA adapter weights (~3-6 MB vs ~3.4 GB for the full UNet)."""
    lora_state = {name: p for name, p in unet.named_parameters() if "lora_" in name}
    torch.save(lora_state, f"{output_dir}/lora.pt")


class DreamBoothModel:
    def __init__(self, device, dtype=torch.float16):
        """
        Load all components. Returns them individually
        so each script has explicit control over what it freezes.
        """
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", dtype=dtype).to(
            device)
        self.vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=dtype).to(device)
        self.scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    def load_finetuned_unet(self, output_dir: str, device, rank: int = 4, alpha: int = 4):
        import os
        if os.path.exists(f"{output_dir}/lora.pt"):
            inject_lora(self.unet, rank=rank, alpha=alpha)
            state = torch.load(f"{output_dir}/lora.pt", map_location=device)
            self.unet.load_state_dict(state, strict=False)
        else:
            state = torch.load(f"{output_dir}/unet.pt", map_location=device)
            self.unet.load_state_dict(state)
        self.unet.to(device)
