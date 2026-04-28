import os
import torch
from diffusers import StableDiffusionPipeline
from model import load_models, load_finetuned_unet, save_unet


def checkpoint(unet, output_dir: str, step: int):
    # Call every N steps in the training loop to snapshot the current UNet weights
    step_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    save_unet(unet, step_dir)
    print(f"Checkpoint saved at step {step} -> {step_dir}")


def validate(unet, prompt: str, output_dir: str, step: int, device: str, dtype=torch.float16, num_images: int = 4):
    os.makedirs(output_dir, exist_ok=True)

    # Load fresh frozen components — we only need the live training UNet, not a new one
    tokenizer, text_encoder, vae, _, scheduler = load_models(device=device, dtype=dtype)

    # Switch to eval mode so dropout/batchnorm don't interfere with image quality
    unet.eval()
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,          # inject the live training UNet instead of the base one
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    # No gradients needed during generation
    with torch.no_grad():
        for i in range(num_images):
            image = pipe(prompt, num_inference_steps=50).images[0]
            # Filename encodes the step so you can track visual progress over time
            image.save(os.path.join(output_dir, f"step_{step}_{i:02d}.jpg"))

    # Must switch back — forgetting this leaves the UNet in eval mode for the rest of training
    unet.train()
    print(f"Validation images saved to {output_dir}")


def run_inference(checkpoint_dir: str, prompt: str, output_dir: str, device: str, dtype=torch.float16, num_images: int = 4):
    # Post-training: loads saved weights from disk rather than a live training UNet
    os.makedirs(output_dir, exist_ok=True)

    tokenizer, text_encoder, vae, _, scheduler = load_models(device=device, dtype=dtype)
    # Overwrite the base UNet with the fine-tuned weights from the checkpoint
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
