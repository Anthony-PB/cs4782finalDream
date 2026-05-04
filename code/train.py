import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import (
    freeze_component,
    DreamBoothModel,
    inject_lora,
    lora_parameters,
    save_lora,
    save_unet,
    DEFAULT_TARGET_MODULES,
)
from data import DreamBoothDataset
from inference import checkpoint, validate
from torch.amp import autocast, GradScaler


# This function is to turn the dataloader Dictionaries into batched tensors
def collate_fn(examples):
    # Extract Stream A
    input_images = [example["instance_image"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    # Append Stream B
    input_images += [example["class_image"] for example in examples]
    prompts += [example["class_prompt"] for example in examples]

    # Stack into a single tensor for the GPU
    pixel_values = torch.stack(input_images)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    return {"pixel_values": pixel_values, "prompts": prompts}


def dreambooth_loss(
    unet,  # the model being trained
    scheduler,  # owns the αt, σt noise tables
    subject_latents,  # the 3-5 dog images encoded by VAE
    subject_encoder_hidden_states,  # embeddings of "a [V] dog"
    prior_latents,  # generated generic dog images encoded by VAE
    prior_encoder_hidden_states,  # embeddings of "a dog"
    device,
    lam: float = 1.0,  # λ in Equation 2 — weights prior loss
):
    """
    L_DB = E[||eps_theta(z_t, t, c) - eps||^2]
         + lambda * E[||eps_theta(z_pr_t, t, c_pr) - eps||^2]
    """
    bsz = subject_latents.shape[0]  # subject images in batch
    bsz_pr = prior_latents.shape[0]  # prior images in batch

    t = torch.randint(
        0, scheduler.config.num_train_timesteps, (bsz,), device=device
    ).long()
    t_pr = torch.randint(
        0, scheduler.config.num_train_timesteps, (bsz_pr,), device=device
    ).long()

    noise = torch.randn_like(subject_latents)
    noise_pr = torch.randn_like(prior_latents)

    z_t = scheduler.add_noise(subject_latents, noise, t)
    # Based on the noise scheduler add noise to the images inputs.
    z_pr_t = scheduler.add_noise(prior_latents, noise_pr, t_pr)

    noise_pred = unet(
        z_t, t, encoder_hidden_states=subject_encoder_hidden_states
    ).sample
    noise_pred_pr = unet(
        z_pr_t, t_pr, encoder_hidden_states=prior_encoder_hidden_states
    ).sample

    loss_subject = F.mse_loss(noise_pred, noise, reduction="mean")
    loss_prior = F.mse_loss(noise_pred_pr, noise_pr, reduction="mean")

    return loss_subject + lam * loss_prior


def forward(vae, text_encoder, tokenizer, pixel_values, prompts, device, dtype):
    """
    Encode a batch of pixel values and prompts into latents and text embeddings.
    Both VAE and text encoder are frozen, so we run under no_grad for efficiency.
    """
    with torch.no_grad():
        # Encode images → latent space and scale by VAE's scaling factor
        latents = vae.encode(pixel_values.to(device, dtype=dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Tokenize prompts and encode to embeddings
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        encoder_hidden_states = text_encoder(text_inputs.input_ids.to(device))[0]

    return latents, encoder_hidden_states


def training_loop(
    instance_dir: str,  # path to your 3-5 subject images
    class_dir: str,  # path to generated prior images
    instance_prompt: str,  # e.g. "a sks dog"
    class_prompt: str,  # e.g. "a dog"
    output_dir: str = "checkpoints",
    validation_dir: str = "validation",
    num_steps: int = 800,
    batch_size: int = 1,
    lr: float = 5e-6,
    lam: float = 1.0,
    checkpoint_every: int = 200,
    validate_every: int = 200,
    device: str = "cuda",
    dtype=torch.float16,
    use_lora: bool = False,
    lora_rank: int = 4,
    lora_alpha: int = 4,
    lora_target_modules: set = DEFAULT_TARGET_MODULES,
    early_stopping_patience: int = 3,  # validation steps without improvement before stopping
):
    model = DreamBoothModel(device=device, dtype=dtype)
    freeze_component(model.text_encoder)
    freeze_component(model.vae)

    if use_lora:
        freeze_component(model.unet)  # freeze base weights; only adapters train
        inject_lora(
            model.unet,
            rank=lora_rank,
            alpha=lora_alpha,
            target_modules=lora_target_modules,
        )
        train_params = lora_parameters(model.unet)
    else:
        # Full fine-tune: cast UNet to fp32 so GradScaler can unscale gradients.
        # Gradient checkpointing keeps activation memory low enough for T4 (16 GB).
        model.unet = model.unet.to(torch.float32)
        model.unet.enable_gradient_checkpointing()
        train_params = list(model.unet.parameters())

    model.unet.train()
    optimizer = torch.optim.AdamW(train_params, lr=lr)
    scaler = GradScaler()  # fp32 trainable params + fp16 autocast in both modes

    dataset = DreamBoothDataset(
        instance_dir=instance_dir,
        class_dir=class_dir,
        instance_prompt=instance_prompt,
        class_prompt=class_prompt,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    dataloader_iter = iter(dataloader)

    print(f"Starting DreamBooth training for {num_steps} steps...")

    best_clip_t = -float('inf')
    best_step = 0
    no_improve_count = 0
    best_dir = os.path.join(output_dir, "best")

    for step in tqdm(range(1, num_steps + 1)):
        # Restart the iterator when exhausted
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        # collate_fn concatenates [instance..., class...] along batch dim
        pixel_values = batch["pixel_values"]  # [2*bsz, C, H, W]
        prompts = batch["prompts"]  # list of 2*bsz strings
        bsz = pixel_values.shape[0] // 2

        # Encode images and prompts into latents and text embeddings
        latents, encoder_hidden_states = forward(
            model.vae,
            model.text_encoder,
            model.tokenizer,
            pixel_values,
            prompts,
            device,
            dtype,
        )

        # Split stream A (subject) and stream B (prior)
        subject_latents = latents[:bsz]
        prior_latents = latents[bsz:]
        subject_encoder_hidden_states = encoder_hidden_states[:bsz]
        prior_encoder_hidden_states = encoder_hidden_states[bsz:]

        with autocast(device, dtype=dtype):
            loss = dreambooth_loss(
                unet=model.unet,
                scheduler=model.scheduler,
                subject_latents=subject_latents,
                subject_encoder_hidden_states=subject_encoder_hidden_states,
                prior_latents=prior_latents,
                prior_encoder_hidden_states=prior_encoder_hidden_states,
                device=device,
                lam=lam,
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % checkpoint_every == 0:
            checkpoint(model.unet, output_dir, step, use_lora=use_lora)

        if step % validate_every == 0:
            clip_t = validate(
                model.unet,
                instance_prompt,
                validation_dir,
                step,
                device,
                dtype,
            )

            if clip_t > best_clip_t:
                best_clip_t = clip_t
                best_step = step
                no_improve_count = 0
                checkpoint(model.unet, best_dir, step=0, use_lora=use_lora)
                print(f"  New best CLIP-T: {best_clip_t:.3f} at step {best_step} — saved to {best_dir}")
            else:
                no_improve_count += 1
                print(f"  No improvement ({no_improve_count}/{early_stopping_patience}), best still step {best_step} ({best_clip_t:.3f})")
                if no_improve_count >= early_stopping_patience:
                    print(f"Early stopping at step {step}.")
                    break

    # Load best checkpoint back into UNet before saving final weights
    if best_step > 0:
        print(f"Restoring best checkpoint from step {best_step} (CLIP-T: {best_clip_t:.3f})")
        if use_lora:
            best_state = torch.load(os.path.join(best_dir, "lora.pt"), map_location=device)
            model.unet.load_state_dict(best_state, strict=False)
        else:
            best_state = torch.load(os.path.join(best_dir, "unet.pt"), map_location=device)
            model.unet.load_state_dict(best_state)

    final_dir = os.path.join(output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    if use_lora:
        save_lora(model.unet, final_dir)
    else:
        save_unet(model.unet, final_dir)
    print(f"Final weights saved to {final_dir}")
    print("Training complete.")


if __name__ == "__main__":
    training_loop(
        instance_dir="data/instance/dog",
        class_dir="data/prior/dog",
        instance_prompt="a sks dog",
        class_prompt="a dog",
        output_dir="checkpoints",
        validation_dir="validation",
        num_steps=800,
    )
