# Running DreamBooth on Google Colab

## What you need before starting
- A Google account with access to [colab.research.google.com](https://colab.research.google.com)
- The notebook file: `dreambooth_colab.ipynb` (from this repo)
- The 7 source files from `code/`: `model.py`, `data.py`, `train.py`, `inference.py`, `generate_prior.py`, `metrics.py`, `evaluate.py`
- 3–5 photos of your subject (JPG or PNG)

---

## Step 1 — Open the notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload `dreambooth_colab.ipynb` from this repo

---

## Step 2 — Enable GPU

1. Click **Runtime → Change runtime type**
2. Choose your GPU based on what you want to run:

| Goal | GPU | Tier |
|---|---|---|
| LoRA only | **T4** | Free |
| Full fine-tune + LoRA comparison | **A100** | Colab Pro+ |

> Full fine-tune requires A100 (40 GB). The fp32 UNet weights + AdamW optimizer state alone exceed T4's 16 GB — it will OOM.

3. Click **Save**

---

## Step 3 — Install dependencies

Run the **"Install dependencies"** cell. Installs diffusers, transformers, and CLIP. Takes ~1 minute.

---

## Step 4 — Configure your subject

Run the **"Configure your subject"** cell and edit the variables before running:

```python
SUBJECT_NAME    = "dog"          # folder name used for data/
TEXT_NAME       = "dog"          # word used in prompts
CLASS_PROMPT    = f"a {TEXT_NAME}"
INSTANCE_PROMPT = f"a sks {TEXT_NAME}"  # leave this line as-is

LORA_RANKS  = [4, 8, 16, 32]    # LoRA bottleneck dimensions to sweep
LORA_LR     = 1e-4               # learning rate for LoRA
FULL_LR     = 5e-6               # learning rate for full fine-tune

lora_only = False  # set True to skip full fine-tune (required on T4)
```

**To switch subjects** (e.g. a cat):
```python
SUBJECT_NAME = "cat"
TEXT_NAME    = "cat"
```

---

## Step 5 — Upload source files

When the **"Upload source files"** cell runs, select all 7 files from the `code/` folder:

```
model.py
data.py
train.py
inference.py
generate_prior.py
metrics.py
evaluate.py
```

---

## Step 6 — Upload your subject photos

When the **"Upload instance images"** cell runs, upload your 3–5 photos. They are saved to `data/<SUBJECT_NAME>/` automatically.

---

## Step 7 — Generate prior images

Run the **"Generate prior images"** cell. Generates 200 generic class images using the frozen base model — these feed the prior-preservation loss during training.

| GPU | Time |
|---|---|
| T4 | ~15 min |
| A100 | ~5 min |

> You can reduce `num_images` to `50` to go faster at the cost of slightly weaker prior preservation.

---

## Step 8 — Train LoRA (rank × LR sweep)

Run the **"Train — LoRA"** cell. Trains low-rank adapter weights for each rank in `LORA_RANKS` at the fixed `LORA_LR`. Each run is 800 steps.

| GPU | Time per run |
|---|---|
| T4 | ~10–15 min |
| A100 | ~4–6 min |

> With the default config (4 ranks), expect ~40–60 min on T4 or ~16–24 min on A100.

Checkpoints saved to `checkpoints/lora_rank_{rank}/step_800/`. Validation images saved to `validation/lora_rank_{rank}/`.

---

## Step 9 — Train full fine-tune (A100 only, optional)

Run the **"Train — Full Fine-Tune"** cell. Trains all UNet weights with gradient checkpointing enabled at the fixed `FULL_LR`. Runs for 1200 steps.

| GPU | Time |
|---|---|
| T4 | ❌ OOM |
| A100 | ~20–25 min |

> Skip this cell (or set `lora_only = True` in the config) if you are on a T4.

Checkpoint saved to `checkpoints/full/step_1200/`. Validation images saved to `validation/full/`.

---

## Step 10 — Run inference (single prompt)

Run the **"Inference"** cell. Edit `INFERENCE_PROMPT` to place your subject in any scene:

```python
INFERENCE_PROMPT = f"{INSTANCE_PROMPT} on the moon"
INFERENCE_PROMPT = f"{INSTANCE_PROMPT} as an oil painting"
INFERENCE_PROMPT = f"{INSTANCE_PROMPT} in a forest"
```

All trained models run on the same prompt. Outputs saved to `outputs/lora_rank_{rank}/` and `outputs/full/`.

---

## Step 11 — Visual comparison

Run the **"Visual comparison"** cell. Displays a grid with one row per trained model (4 images per row) so you can compare subject fidelity and scene quality across all rank and LR combinations side by side.

---

## Step 12 — Metrics comparison (single prompt)

Run the **"Metrics comparison"** cell. Computes and prints a table for each model:

| Metric | What it measures |
|---|---|
| CLIP-I | Subject fidelity — cosine similarity between CLIP embeddings of generated vs real photos |
| DINO | Subject fidelity — feature-level identity match using DINO ViT features |
| CLIP-T | Prompt fidelity — how well the generated image matches the text prompt |

Higher is better for all three. Results are plotted as a bar chart alongside the paper's reference values.

---

## Step 13 — Multi-prompt inference and averaged metrics

Run the **"Inference and Metrics with Multiple Prompts (Averaged)"** cell followed by the **"Averaged metrics comparison"** cell. Each model generates images for 5 different scene prompts; metrics are averaged across all prompts for a more stable score.

---

## Step 14 — Download outputs

From the Colab file browser (left sidebar, folder icon):

- `outputs/` — all inference images organised by model and LR
- `checkpoints/` — saved model weights
- `validation/` — per-step validation images from training

Right-click any folder → **Download** to get everything at once.

---

## Notes

- **Colab disconnects wipe all files.** If your session drops mid-run you will need to re-upload files and restart from the beginning.
- **Free Colab has GPU time limits.** Run prior generation and all training cells in one session without leaving the tab idle.
- **The LoRA rank sweep is the main experiment.** If you are on T4, set `lora_only = True` and reduce `LORA_RANKS` (e.g. `[4, 16]`) to keep total runtime under the free GPU limit.
- LoRA checkpoints are ~3–6 MB each (`lora.pt`). Full fine-tune checkpoints are ~3.4 GB each (`unet.pt`).
