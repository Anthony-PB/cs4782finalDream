import os
import torch
from diffusers import StableDiffusionPipeline
from model import load_models, load_finetuned_unet, save_unet


def checkpoint(unet, output_dir: str, step: int):
    step_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    save_unet(unet, step_dir)
    print(f"Checkpoint saved at step {step} -> {step_dir}")


def validate(unet, prompt: str, output_dir: str, step: int, device: str, dtype=torch.float16, num_images: int = 4):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, text_encoder, vae, _, scheduler = load_models(device=device, dtype=dtype)

    # swap in the current fine-tuned unet
    unet.eval()
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    with torch.no_grad():
        for i in range(num_images):
            image = pipe(prompt, num_inference_steps=50).images[0]
            image.save(os.path.join(output_dir, f"step_{step}_{i:02d}.jpg"))

    unet.train()
    print(f"Validation images saved to {output_dir}")


def run_inference(checkpoint_dir: str, prompt: str, output_dir: str, device: str, dtype=torch.float16, num_images: int = 4):
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, text_encoder, vae, _, scheduler = load_models(device=device, dtype=dtype)
    unet = load_finetuned_unet(checkpoint_dir, device=device, dtype=dtype)

    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    with torch.no_grad():
        for i in range(num_images):
            image = pipe(prompt, num_inference_steps=50).images[0]
            image.save(os.path.join(output_dir, f"{i:02d}.jpg"))

    print(f"Saved {num_images} images to {output_dir}")


if __name__ == "__main__":
    run_inference(
        checkpoint_dir="checkpoints/step_800",
        prompt="a sks dog in the park",
        output_dir="outputs/inference",
        device="cuda",
        num_images=4,
    )
