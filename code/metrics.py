import torch
import torch.nn.functional as F
import clip
from torchvision import transforms
from PIL import Image
import os

# Cache models to avoid reloading
_clip_model = None
_clip_preprocess = None
_dino_model = None


def load_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32")
        _clip_model.eval()
    return _clip_model, _clip_preprocess


def load_dino():
    global _dino_model
    if _dino_model is None:
        _dino_model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
        _dino_model.eval()
    return _dino_model


def compute_clip_i(generated_images, real_images):
    """
    Compute CLIP-I: average pairwise cosine similarity between CLIP embeddings of generated and real images.
    generated_images: list of PIL Images
    real_images: list of PIL Images
    """
    model, preprocess = load_clip()

    def get_embeddings(images):
        embeds = []
        for img in images:
            img_tensor = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                embed = model.encode_image(img_tensor)
            embeds.append(embed.squeeze())
        return embeds

    gen_embeds = get_embeddings(generated_images)
    real_embeds = get_embeddings(real_images)

    similarities = []
    for g in gen_embeds:
        for r in real_embeds:
            sim = F.cosine_similarity(g, r, dim=0).item()
            similarities.append(sim)

    return sum(similarities) / len(similarities) if similarities else 0.0


def compute_dino(generated_images, real_images):
    """
    Compute DINO: average pairwise cosine similarity between DINO ViT-S/16 embeddings of generated and real images.
    """
    model = load_dino()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def get_embeddings(images):
        embeds = []
        for img in images:
            img_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                embed = output[:, 0, :]  # CLS token
            embeds.append(embed.squeeze())
        return embeds

    gen_embeds = get_embeddings(generated_images)
    real_embeds = get_embeddings(real_images)

    similarities = []
    for g in gen_embeds:
        for r in real_embeds:
            sim = F.cosine_similarity(g, r, dim=0).item()
            similarities.append(sim)

    return sum(similarities) / len(similarities) if similarities else 0.0


def compute_clip_t(generated_images, prompts):
    """
    Compute CLIP-T: average cosine similarity between prompt CLIP embeddings and image CLIP embeddings.
    prompts: str or list of str (if list, must match length of generated_images)
    """
    model, preprocess = load_clip()

    if isinstance(prompts, str):
        prompts = [prompts] * len(generated_images)
    elif len(prompts) != len(generated_images):
        raise ValueError("prompts list must match generated_images length")

    text_embeds = []
    for prompt in prompts:
        tokens = clip.tokenize([prompt])
        with torch.no_grad():
            embed = model.encode_text(tokens)
        text_embeds.append(embed.squeeze())

    img_embeds = []
    for img in generated_images:
        img_tensor = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            embed = model.encode_image(img_tensor)
        img_embeds.append(embed.squeeze())

    similarities = []
    for img_embed, text_embed in zip(img_embeds, text_embeds):
        sim = F.cosine_similarity(img_embed, text_embed, dim=0).item()
        similarities.append(sim)

    return sum(similarities) / len(similarities) if similarities else 0.0
