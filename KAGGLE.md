# Run RUDRA Training on Kaggle

## Step 1: Create a Kaggle Notebook

1. Go to https://www.kaggle.com/notebooks
2. Click "New Notebook"
3. Set these settings (right sidebar → "Settings" tab):

| Setting | Value |
|---------|-------|
| **Accelerator** | `T4 x2` (or `P100` if T4 unavailable) |
| **Internet** | `On` |
| **Persistent storage** | `Off` (default) |
| **GPU RAM** | `High` (16GB) |

## Step 2: Paste the Script

1. In the first code cell, paste the entire contents of `kaggle_run.py`
2. Before running, update the `REPO_URL` on line 27 to your actual GitHub repo URL:
   ```python
   REPO_URL = "https://github.com/YOUR_USERNAME/rudra"
   ```
3. Or, if you want to upload files directly instead of cloning:
   - Comment out the git clone block (lines 26-33)
   - Upload the `rudra/` directory via Kaggle's "Upload Data" button
   - Set `os.chdir("/kaggle/working/rudra")` to the upload path

## Step 3: Run

1. Click "Run All" (or run cell by cell with Shift+Enter)
2. Expected time: **~3-4 hours** for a full 3-stage run on T4

| Stage | Data | ETA |
|-------|------|-----|
| Data generation | A-G + DPO | ~30 min (streaming from HF) |
| Stage 1 SFT | B+C+E (27K, 3 epochs, LoRA r=32) | ~2 hours |
| Stage 2 Behavior Lock | A+D (3.6K, 3 epochs, LoRA r=16) | ~30 min |
| Stage 3 DPO | 1.5K pairs, 1 epoch | ~30 min |
| Eval | Baseline Qwen + RUDRA | ~30 min |

## Step 4: Download Results

When the notebook finishes:

1. Click **File → Export as ZIP** (top menu)
2. This downloads `kaggle/working/rudra/output/` (checkpoints) and `eval_results/` (comparison tables)
3. Or use the Kaggle API:
   ```python
   # In a new cell at the end:
   import shutil
   shutil.make_archive("/kaggle/working/rudra_output", "zip", "/kaggle/working/rudra/output")
   shutil.make_archive("/kaggle/working/rudra_eval", "zip", "/kaggle/working/rudra/eval_results")
   ```

## Alternative: Lightning AI (80 free GPU hours/month)

If Kaggle has no GPU slots:

1. Go to https://lightning.ai
2. Create account (phone verification for 80 free hours)
3. Create a new "App" with GPU
4. Upload repo or clone it
5. Install deps:
   ```bash
   pip install torch transformers datasets accelerate peft trl bitsandbytes sentencepiece evalplus wandb
   ```
6. Run:
   ```bash
   python src/generate_data.py
   python -c "from src.generate_data import generate_dpo_pairs; generate_dpo_pairs('data/dpo_pairs')"
   python src/train.py --config configs/train_config.yaml --start-stage 1
   ```

## Expected Output

After training completes:

```
output/
├── stage1/final/      # LoRA r=32 SFT checkpoint (~1.5B params)
├── stage2/final/      # LoRA r=16 behavior lock checkpoint
└── stage3/final/      # DPO-optimized checkpoint (final model)

eval_results/
├── baseline_qwen.json     # Qwen2.5-Coder-1.5B eval scores
├── rudra_stage3.json      # RUDRA eval scores
├── COMPARISON.md          # Markdown comparison table
└── training_summary.json  # Time and status log
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "CUDA out of memory" | Try `T4 x2` instead of `T4 x1`, or reduce batch size in `train_config.yaml` |
| "HF datasets download slow" | Ensure Internet is `On` in Notebook settings |
| "Session expires (9h limit)" | Use `--start-stage 2` to resume from Stage 2 |
| "Kaggle no GPU slots" | Try Lightning AI (80 free hours) or Colab (less reliable) |