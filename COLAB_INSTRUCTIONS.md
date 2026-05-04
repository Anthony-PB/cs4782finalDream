# Running DreamBooth on Google Colab

## What you need before starting
- A Google account with access to [colab.research.google.com](https://colab.research.google.com)
- The notebook file: `dreambooth_colab.ipynb` (from this repo)
- The 6 source files from `code/`: `model.py`, `data.py`, `train.py`, `inference.py`, `generate_prior.py`, `metrics.py`
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

Run **cell 1** (`pip install`). Installs diffusers, transformers, and CLIP. Takes ~1 minute.

---

## Step 4 — Configure your subject

Run **cell 3** and edit the variables before running:

```python
SUBJECT_NAME    = "killian"               # any name, used for the data folder
CLASS_NAME      = "white-middle-aged-man" # describes the general category
CLASS_PROMPT    = "a white middle aged man"
INSTANCE_PROMPT = f"a sks {CLASS_PROMPT}" # leave this line as-is
```

**To switch subjects** (e.g. a cat):
```python
CLASS_NAME   = "cat"
CLASS_PROMPT = "a cat"
```

---

## Step 5 — Upload source files

When the **"Upload source files"** cell runs, select all 6 files from the `code/` folder:

```
model.py
data.py
train.py
inference.py
generate_prior.py
metrics.py
```

---

## Step 6 — Upload your subject photos

When the **"Upload instance images"** cell runs, upload your 3–5 photos. They are saved to `data/<SUBJECT_NAME>/` automatically.

---

## Step 7 — Generate prior images

Run the **"Generate prior images"** cell. Generates 200 generic class images used for prior-preservation loss.

| GPU | Time |
|---|---|
| T4 | ~15 min |
| A100 | ~5 min |

> You can reduce `num_images` to `50` to go faster at the cost of slightly weaker prior preservation.

---

## Step 8 — Train full fine-tune (A100 only)

Run the **"Train — Full Fine-Tune"** cell. Trains all UNet weights with gradient checkpointing enabled to fit in GPU memory.

| GPU | Time |
|---|---|
| T4 | ❌ OOM |
| A100 | ~20–25 min |

Checkpoints saved every 200 steps to `checkpoints/full/`. Validation images saved to `validation/full/`.

---

## Step 9 — Train LoRA

Run the **"Train — LoRA"** cell. Trains only small adapter weights (~3–6 MB).

| GPU | Time |
|---|---|
| T4 | ~10–15 min |
| A100 | ~4–6 min |

Checkpoints saved to `checkpoints/lora/`. Validation images saved to `validation/lora/`.

---

## Step 10 — Run inference

Run the **"Inference"** cell. Edit `INFERENCE_PROMPT` to place your subject in any scene:

```python
INFERENCE_PROMPT = f"{INSTANCE_PROMPT} on the moon"
INFERENCE_PROMPT = f"{INSTANCE_PROMPT} as an oil painting"
INFERENCE_PROMPT = f"{INSTANCE_PROMPT} in a forest"
```

Both models run on the same prompt. Outputs saved to `outputs/full/` and `outputs/lora/`.

---

## Step 11 — Visual comparison

Run the **"Visual comparison"** cell. Displays a 2×4 grid — full fine-tune on top, LoRA on bottom — so you can compare subject fidelity and scene quality side by side.

---

## Step 12 — Metrics comparison

Run the **"Metrics comparison"** cell. Computes and prints a table:

| Metric | What it measures |
|---|---|
| CLIP-I | Subject fidelity — similarity to your real photos |
| DINO | Subject fidelity — feature-level identity match |
| CLIP-T | Prompt fidelity — how well the scene matches the text |

Higher is better for all three.

---

## Step 13 — Download outputs

From the Colab file browser (left sidebar, folder icon):

- `outputs/full/` — full fine-tune images
- `outputs/lora/` — LoRA images
- `checkpoints/` — saved model weights
- `validation/` — per-step validation images from training

Right-click any folder → **Download** to get everything at once.

---

## Notes

- **Colab disconnects wipe all files.** If your session drops mid-run you will need to re-upload files and restart from the beginning.
- **Free Colab has GPU time limits.** Run prior generation and both training cells in one session without leaving the tab idle.
- The final weights are always at `checkpoints/lora/final/lora.pt` (~3–6 MB) and `checkpoints/full/final/unet.pt` (~3.4 GB), regardless of when early stopping triggered.
