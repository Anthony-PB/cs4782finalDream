import os
import torch
from diffusers import StableDiffusionPipeline
from model import DreamBoothModel
from metrics import compute_clip_i, compute_dino, compute_clip_t
from PIL import Image
import csv


def load_images_from_folder(folder_path):
    images = []
    for file in os.listdir(folder_path):
        if file.endswith(".jpg"):
            img = Image.open(os.path.join(folder_path, file)).convert("RGB")
            images.append(img)
    return images


def main():
    checkpoint_dir = "checkpoints/step_800"  # Update this path as needed
    output_dir = "results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    num_images = 10  # Number of images to generate per subject

    # Get list of subjects
    dataset_dir = "data/dreambooth-dataset"
    subjects = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    # Load the fine-tuned model
    model = DreamBoothModel(device=device, dtype=dtype)
    model.load_finetuned_unet(checkpoint_dir, device=device, dtype=dtype)

    pipe = StableDiffusionPipeline(
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        unet=model.unet,
        scheduler=model.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)

    results = []
    for subject in subjects:
        prompt = f"a sks {subject}"
        real_folder = os.path.join(dataset_dir, subject)
        real_images = load_images_from_folder(real_folder)
        if not real_images:
            print(f"No images found for {subject}, skipping.")
            continue

        # Generate images
        generated_images = []
        with torch.no_grad():
            for i in range(num_images):
                image = pipe(prompt, num_inference_steps=50).images[0]
                generated_images.append(image)

        # Compute metrics
        clip_i = compute_clip_i(generated_images, real_images)
        dino = compute_dino(generated_images, real_images)
        clip_t = compute_clip_t(generated_images, prompt)

        results.append(
            {"subject": subject, "clip_i": clip_i, "dino": dino, "clip_t": clip_t}
        )

        print(f"{subject}: CLIP-I={clip_i:.3f}, DINO={dino:.3f}, CLIP-T={clip_t:.3f}")

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["subject", "clip_i", "dino", "clip_t"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Compute and save averages
    if results:
        avg_clip_i = sum(r["clip_i"] for r in results) / len(results)
        avg_dino = sum(r["dino"] for r in results) / len(results)
        avg_clip_t = sum(r["clip_t"] for r in results) / len(results)

        print(
            f"Averages: CLIP-I={avg_clip_i:.3f}, DINO={avg_dino:.3f}, CLIP-T={avg_clip_t:.3f}"
        )

        with open(csv_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["average", avg_clip_i, avg_dino, avg_clip_t])


if __name__ == "__main__":
    main()
