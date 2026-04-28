import os
import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from model import DreamBoothModel, freeze_component


def generate_prior_images(
    class_prompt: str,
    output_dir: str,
    num_images: int = 200,
    device: str = "cuda",
    dtype = torch.float16
):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Load all components via model.py
    model = DreamBoothModel(device=device, dtype=dtype)

    # Step 2: Freeze everything — why all components here
    freeze_component(model.text_encoder)
    freeze_component(model.vae)
    freeze_component(model.unet)

    # Step 3: Wrap into pipeline for easy inference
    pipe = StableDiffusionPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        unet=model.unet,
        scheduler=model.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False
    ).to(device)

    # Step 4: Generate images
    print(f"Generating {num_images} prior images for prompt: '{class_prompt}'")

    with torch.no_grad():
        for i in tqdm(range(num_images)):
            
            generator = torch.Generator(device=device).manual_seed(i)

            image = pipe(
                class_prompt,
                num_inference_steps=50,
                generator=generator
            ).images[0]

            image.save(os.path.join(output_dir, f"{i:04d}.jpg"))

    print(f"Saved {num_images} images to {output_dir}")


if __name__ == "__main__":
    generate_prior_images(
        class_prompt="a dog",
        output_dir="data/prior/dog",
        num_images=200
    )
