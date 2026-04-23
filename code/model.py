# model.py
import torch
from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_ID = "runwayml/stable-diffusion-v1-5"

def load_models(device, dtype=torch.float16):
    """
    Load all components. Returns them individually
    so each script has explicit control over what it freezes.
    """
    # TODO 1: Load tokenizer

    # TODO 2: Load text encoder
    
    # TODO 3: Load VAE

    # TODO 4: Load UNet — this is the one being fine-tuned

    # TODO 5: Load scheduler

    return tokenizer, text_encoder, vae, unet, scheduler


def freeze_component(component):
    """
    TODO 6: One clean reusable function to freeze any component.
    Why is this better than writing requires_grad_(False) 
    everywhere?
    """
    pass


def save_unet(unet, output_dir: str):
    """
    TODO 7: Save only the UNet weights.
    What format should we use and why?
    """
    pass


def load_finetuned_unet(output_dir: str, device, dtype=torch.float16):
    """
    TODO 8: Load base UNet then overwrite with fine-tuned weights.
    Why do we load the base first rather than just loading 
    the saved weights directly?
    """
    pass