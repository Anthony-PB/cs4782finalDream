### Introduction

This repo re-implements DreamBooth (Ruiz et al., 2023), a method that fine-tunes a text-to-image diffusion model on just 3–5 images of a specific subject using a unique token, enabling generation of that subject in novel scenes and contexts.

### Chosen Result

We targeted the quantitative evaluation from Table 2 of the paper, which benchmarks subject fidelity (DINO, CLIP-I) and prompt fidelity (CLIP-T) against Textual Inversion and real images, the core evidence for DreamBooth's superiority over prior personalization methods.

| Method                               |     DINO↑ |   CLIP-I↑ |   CLIP-T↑ |
| ------------------------------------ | --------: | --------: | --------: |
| Real Images                          |     0.774 |     0.885 |       N/A |
| DreamBooth (Stable Diffusion)        |     0.668 |     0.803 |     0.305 |
| Textual Inversion (Stable Diffusion) |     0.569 |     0.780 |     0.255 |

**Table 2 (from paper).** Subject fidelity (DINO, CLIP-I) and prompt fidelity (CLIP-T), the targets we reproduce.

### GitHub Contents

`code/`, all training, inference, and evaluation scripts; `data/`, official DreamBooth dataset (30 subjects) plus our added `killian/` subject; `results/`, metric outputs across LoRA ranks; `poster/` and `report/`, project writeups.

### Re-implementation Details

We fine-tune only the UNet of Stable Diffusion v1.5 using LoRA adapters (ranks r ∈ {4, 8, 16, 32}, ~0.3M params) injected into all attention projections, trained for 800 steps with AdamW (lr=5e-6) on the prior preservation loss using the special token `"sks"`, and evaluate with DINO, CLIP-I, and CLIP-T on the official 30-subject DreamBooth dataset plus our added human subject.

### Reproduction Steps

**GPU requirements:**
- LoRA fine-tune: confirmed on a free Colab T4 (16 GB); lower VRAM may also work
- Full fine-tune: confirmed on a Colab Pro+ A100 (40 GB); runs out of memory on a T4 (16 GB)

Install dependencies with `pip install -r requirements.txt`, then run `code/generate_prior.py` to generate class images, `code/train.py` to fine-tune, and `code/inference.py` to generate images from a saved checkpoint. Edit the `__main__` block in each script to set subject paths, prompts, and hyperparameters.

For step-by-step instructions on running everything through the provided Colab notebook, see `COLAB_INSTRUCTIONS.md`.

### Results/Insights

Our LoRA re-implementation matches or exceeds the paper's SD baseline on object subjects (Dog6: CLIP-I 0.865 vs. 0.803, DINO 0.770 vs. 0.668) while achieving near-paper prompt fidelity; human face identity (Killian: CLIP-I 0.590, DINO 0.316) remains the primary shortfall.

| Subject | Method | DINO↑ | CLIP-I↑ | CLIP-T↑ |
| --------------- | ----------- | ----: | ------: | ------: |
| Dog6 Beach | LoRA r=4 | 0.770 | 0.865 | 0.289 |
| Dog6 Beach | LoRA r=32 | 0.797 | 0.881 | 0.281 |
| Dog6 Beach | Paper (SD) | 0.668 | 0.803 | 0.305 |
| Killian (human) | LoRA r=4 | 0.316 | 0.590 | 0.301 |
| Killian (human) | LoRA r=32 | 0.385 | 0.619 | 0.301 |
| Killian (human) | Paper (SD) | 0.668 | 0.803 | 0.305 |

### Conclusion

LoRA fine-tuning on publicly available SD v1.5 matches the paper's full fine-tune baseline on object subjects while cutting trainable parameters by ~3000×, though quantitative metrics understate qualitative differences between ranks, highlighting the limits of automated evaluation for generative models.

### References

- Ruiz, Nataniel, et al. 'DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation'. arXiv [Cs.CV], 2023, arxiv.org/abs/2208.12242. arXiv.
- Hu, Edward J., et al. 'LoRA: Low-Rank Adaptation of Large Language Models'. arXiv [Cs.CL], 2021, arxiv.org/abs/2106.09685. arXiv.
- "Introduction to Diffusers." Diffusion Course · Hugging Face, huggingface.co/learn/diffusion-course/unit1/2.

### Acknowledgements

This project was completed as part of Cornell University's CS 4782 (Deep Learning), Spring 2026. Group members: Alex McGowan (acm355), Anthony Paredes-Bautista (ap2357), August Ehrlich (ae427), and Nathnael Tesfaw (nbt26).
