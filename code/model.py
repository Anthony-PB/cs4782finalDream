# Purpose: loads/freezes/saves all of the model components
import torch
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


class DreamBoothModel:
    def __init__(self, device, dtype=torch.float16):
        """
        Load all components. Returns them individually
        so each script has explicit control over what it freezes.
        """
        self.tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder", torch_dtype=dtype).to(
            device)
        self.vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype).to(device)
        self.unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet", torch_dtype=dtype).to(device)
        self.scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    def load_finetuned_unet(self, output_dir: str, device):
        """
        Load base UNet then overwrite with fine-tuned weights.
        """
        self.unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet")
        save = torch.load(f"{output_dir}/unet.pt")
        self.unet.load_state_dict(save)
        self.unet.to(device)
