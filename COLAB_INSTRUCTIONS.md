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
2. Set **Hardware accelerator** to **T4 GPU**
3. Click **Save**

---

## Step 3 — Install dependencies

Run **cell 1** (the `pip install` cell). This installs diffusers, transformers, and CLIP. Takes ~1 minute.

---

## Step 4 — Configure your subject

Run **cell 3** (the config cell) and edit the four variables at the top before running it:

```python
SUBJECT_NAME    = "killian"               # any name, used for the data folder
CLASS_NAME      = "white-middle-aged-man" # describes the general category
CLASS_PROMPT    = "a white middle aged man"
INSTANCE_PROMPT = f"a sks {CLASS_PROMPT}" # leave this line as-is
```

**To switch subjects** (e.g. a cat), just change:
```python
CLASS_NAME   = "cat"
CLASS_PROMPT = "a cat"
```

You can also control whether LoRA is used:
```python
USE_LORA = False  # default — full UNet fine-tune, slower but stronger
USE_LORA = True   # LoRA adapters only, faster and saves only ~3-6 MB
```

Run the **LoRA check cell** directly below to confirm the mode before training.

The cell prints the derived folder paths, prompts, and LoRA mode so you can confirm everything looks right before continuing.

---

## Step 5 — Upload source files

When the **"Upload source files"** cell runs, a file picker will appear. Select all 6 files from the `code/` folder:

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

When the **"Upload instance images"** cell runs, upload your 3–10 photos of the subject. They will be saved to `data/<SUBJECT_NAME>/` automatically.

---

## Step 7 — Generate prior images (~15 min)

Run the **"Generate prior images"** cell. This generates 200 generic images of the class (e.g. "a white middle aged man") using the base Stable Diffusion model. These are used for prior-preservation loss during training so the model doesn't forget the broader class.

> You can reduce `num_images` to `50` if you want to go faster and don't mind slightly weaker prior preservation.

---

## Step 8 — Train (~10-15min)

Run the **"Train"** cell. It runs 800 steps of DreamBooth with LoRA and saves checkpoints every 200 steps to `checkpoints/`. Validation images are saved to `validation/` every 200 steps so you can track progress.

---

## Step 9 — Run inference

Run the **"Run inference"** cell. Edit `INFERENCE_PROMPT` to place your subject in any scene:

```python
INFERENCE_PROMPT = f"{INSTANCE_PROMPT} on the moon"
INFERENCE_PROMPT = f"{INSTANCE_PROMPT} as an oil painting"
INFERENCE_PROMPT = f"{INSTANCE_PROMPT} in a forest"
```

4 images are generated and displayed inline, and saved to `outputs/`.

---

## Step 10 — Download outputs

After inference, download your results from the Colab file browser on the left sidebar:

1. Click the **folder icon** (Files) in the left panel
2. Navigate to `outputs/`
3. Right-click any image → **Download**, or right-click the `outputs/` folder → **Download**

To also download checkpoints or validation images, do the same for `checkpoints/` and `validation/`.

---

## Notes

- **Colab disconnects wipe all files.** If your session disconnects mid-run, you will need to re-upload files and restart from the beginning.
- **Free Colab has GPU time limits.** If you get disconnected, consider running prior generation and training in the same session without leaving the tab idle.
- The LoRA adapter weights are saved at `checkpoints/step_800/lora.pt` (~3–6 MB). These are what actually encodes your subject — much smaller than the full model.
