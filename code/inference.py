import os
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from model import DreamBoothModel, save_lora, save_unet, MODEL_ID
from metrics import compute_clip_t


def checkpoint(unet, output_dir: str, step: int, use_lora: bool = True):
    # Call every N steps in the training loop to snapshot the current UNet weights
    step_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)
    if use_lora:
        save_lora(unet, step_dir)
    else:
        save_unet(unet, step_dir)
    print(f"Checkpoint saved at step {step} -> {step_dir}")


def validate(
    unet,
    prompt: str,
    output_dir: str,
    step: int,
    device: str,
    dtype=torch.float16,
    num_images: int = 4,
    compute_metrics: bool = False,
):
    os.makedirs(output_dir, exist_ok=True)

    # If full-finetune kept UNet in fp32 for stability, cast to inference dtype
    # so the pipeline runs cleanly with the rest of the (fp16) components.
    original_unet_dtype = next(unet.parameters()).dtype
    needs_cast = original_unet_dtype != dtype
    if needs_cast:
        unet.to(dtype)

    model = DreamBoothModel(device=device, dtype=dtype)
    # patch in the unet parameter
    model.unet = unet

    # Switch to eval mode so dropout/batchnorm don't interfere with image quality
    model.unet.eval()
    # DDPMScheduler is for training only — use DDIM for fast inference
    inference_scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        unet=model.unet,  # inject the live training UNet instead of the base one
        scheduler=inference_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    # No gradients needed during generation
    with torch.no_grad():
        images = []
        for i in range(num_images):
            image = pipe(prompt, num_inference_steps=50).images[0]
            # Filename encodes the step so you can track visual progress over time
            image.save(os.path.join(output_dir, f"step_{step}_{i:02d}.jpg"))
            images.append(image)

        if compute_metrics:
            clip_t = compute_clip_t(images, prompt)
            print(f"Validation CLIP-T at step {step}: {clip_t:.3f}")

    # Restore training dtype so the next training step can keep using GradScaler
    if needs_cast:
        unet.to(original_unet_dtype)
    # Must switch back — forgetting this leaves the UNet in eval mode for the rest of training
    unet.train()
    print(f"Validation images saved to {output_dir}")


def run_inference(
    checkpoint_dir: str,
    prompt: str,
    output_dir: str,
    device: str,
    dtype=torch.float16,
    num_images: int = 4,
    compute_metrics: bool = False,
):
    # Post-training: loads saved weights from disk rather than a live training UNet
    os.makedirs(output_dir, exist_ok=True)

    model = DreamBoothModel(device=device, dtype=dtype)
    # Overwrite the base UNet with the fine-tuned weights from the checkpoint
    model.load_finetuned_unet(checkpoint_dir, device=device)

    inference_scheduler = DDIMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        unet=model.unet,
        scheduler=inference_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    with torch.no_grad():
        images = []
        for i in range(num_images):
            image = pipe(prompt, num_inference_steps=50).images[0]
            image.save(os.path.join(output_dir, f"{i:02d}.jpg"))
            images.append(image)

        if compute_metrics:
            clip_t = compute_clip_t(images, prompt)
            print(f"Inference CLIP-T: {clip_t:.3f}")

    print(f"Saved {num_images} images to {output_dir}")
    return images


if __name__ == "__main__":
    run_inference(
        checkpoint_dir="checkpoints/step_800",
        prompt="a sks dog in the park",
        output_dir="outputs/inference",
        device="cuda",
        num_images=4,
    )
