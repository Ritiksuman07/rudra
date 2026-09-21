"""
RUDRA — Kaggle Notebook (Dataset Upload Method)
================================================
Step 1: Upload rudra_for_kaggle.zip as a Kaggle Dataset
Step 2: Paste this into a Kaggle Notebook, set GPU + Internet ON
Step 3: Run All

How to upload:
  kaggle.com → Datasets → New Dataset → Upload File → rudra_for_kaggle.zip
  Name it "rudra-repo"
  The files will appear at /kaggle/input/rudra-repo/
"""
import os, sys, subprocess, time, shutil

START_TIME = time.time()
def log(msg): print(f"[{time.strftime('%H:%M:%S', time.gmtime(time.time()-START_TIME))}] {msg}"); sys.stdout.flush()

# ── 1. Copy repo from uploaded dataset ──
SRC = "/kaggle/input/rudra-repo"
DST = "/kaggle/working/rudra"

if os.path.exists(DST):
    shutil.rmtree(DST)

# Find the actual source — might be nested
if os.path.exists(f"{SRC}/rudra_for_kaggle.zip"):
    log("Extracting uploaded ZIP...")
    import zipfile
    with zipfile.ZipFile(f"{SRC}/rudra_for_kaggle.zip") as zf:
        zf.extractall("/kaggle/working/")
elif os.path.exists(f"{SRC}/src/generate_data.py"):
    shutil.copytree(SRC, DST)
else:
    # Try common Kaggle upload structures
    for candidate in [SRC, f"{SRC}/rudra", f"{SRC}/rudra-main/rudra-main", f"{SRC}/rudra-main"]:
        if os.path.exists(f"{candidate}/src/generate_data.py"):
            shutil.copytree(candidate, DST)
            break
    else:
        raise FileNotFoundError(f"Could not find RUDRA source. Upload the ZIP as a dataset named 'rudra-repo'. Contents of {SRC}: {os.listdir(SRC)}")

os.chdir(DST)
log(f"Copied to {DST}")
log(f"Contents: {os.listdir()}")

# ── 2. Install deps ──
log("Installing dependencies...")
subprocess.run(["pip", "install", "-q",
    "torch", "transformers", "datasets", "accelerate",
    "peft", "trl", "bitsandbytes", "sentencepiece",
    "evalplus", "wandb", "tensorboard", "pyyaml"], check=True)

# ── 3. Generate data ──
log("Generating training data...")
from src.generate_data import generate_all_data, generate_dpo_pairs
results = generate_all_data()
dpo_path = generate_dpo_pairs("data/dpo_pairs")
log(f"Data ready. DPO at {dpo_path}")

# ── 4. Train 3 stages ──
from src.train import RudraTrainingConfig, setup_tokenizer
from src.train import stage_1_sft, stage_2_behavior_lock, stage_3_dpo

config = RudraTrainingConfig(config_path="configs/train_config.yaml")
tokenizer = setup_tokenizer(config.model_name)

log("Stage 1: SFT on B+C+E..."); stage_1_sft(config, tokenizer); log("Stage 1 done")
log("Stage 2: Behavior Lock on A+D..."); stage_2_behavior_lock(config, tokenizer); log("Stage 2 done")
log("Stage 3: DPO..."); stage_3_dpo(config, tokenizer); log("Stage 3 done")

# ── 5. Eval ──
log("Evaluating...")
os.makedirs("eval_results", exist_ok=True)
subprocess.run(["python", "-m", "eval.compare_table", "--rudra", "./output/stage3/final", "--output-dir", "eval_results"], check=False)

hours = (time.time() - START_TIME) / 3600
log(f"Done in {hours:.1f}h. Results in eval_results/")
log("Download: File → Export as ZIP to get output/ and eval_results/")