"""Copy this entire cell into a new Kaggle Notebook and run it."""
import os, sys, subprocess, time, shutil, zipfile

t0 = time.time()
def log(m): print(f"[{time.strftime('%H:%M:%S',time.gmtime(time.time()-t0))}] {m}"); sys.stdout.flush()

# 1. Find the uploaded dataset
SRC = "/kaggle/input"
for root, dirs, files in os.walk(SRC):
    for f in files:
        if f.endswith(".zip") and "rudra" in f.lower():
            log(f"Found ZIP at {os.path.join(root, f)}")
            with zipfile.ZipFile(os.path.join(root, f)) as zf:
                zf.extractall("/kaggle/working/")
            break
    else:
        continue
    break

# Find src/generate_data.py anywhere in working
import glob
candidates = glob.glob("/kaggle/working/**/src/generate_data.py", recursive=True)
if candidates:
    wd = os.path.dirname(os.path.dirname(candidates[0]))
else:
    # Maybe dataset was extracted as a folder
    for d in os.listdir("/kaggle/working/"):
        if os.path.isdir(f"/kaggle/working/{d}/src"):
            wd = f"/kaggle/working/{d}"
            break
    else:
        raise FileNotFoundError(f"src/generate_data.py not found. List: {os.listdir('/kaggle/working/')}")

os.chdir(wd)
log(f"WD: {wd} | Files: {os.listdir()}")

# 2. Install deps
log("Installing...")
subprocess.run(["pip","install","-q","torch","transformers","datasets","accelerate","peft","trl","bitsandbytes","sentencepiece","pyyaml","evalplus"], check=True)

# 3. Generate data + train
log("Gen data...")
from src.generate_data import generate_all_data, generate_dpo_pairs
generate_all_data(); generate_dpo_pairs("data/dpo_pairs")

log("Stage 1 SFT...")
from src.train import RudraTrainingConfig, setup_tokenizer, stage_1_sft, stage_2_behavior_lock, stage_3_dpo
cfg = RudraTrainingConfig("configs/train_config.yaml"); tok = setup_tokenizer(cfg.model_name)
stage_1_sft(cfg, tok)

log("Stage 2 Behavior Lock...")
stage_2_behavior_lock(cfg, tok)

log("Stage 3 DPO...")
stage_3_dpo(cfg, tok)

log(f"DONE in {(time.time()-t0)/3600:.1f}h. Export output/ and eval_results/ via File > Export as ZIP")