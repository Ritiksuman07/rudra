"""
RUDRA — Kaggle Notebook Script
===============================
1. Clone repo, install deps
2. Generate all data (A-G + DPO)
3. Run 3-stage training
4. Run baseline + RUDRA eval
5. Save results

How to use:
1. Create a new Kaggle Notebook (https://kaggle.com)
2. Set Accelerator = "T4 x2" (or P100)
3. Set Internet = "On"
4. Set GPU RAM = "High"
5. Paste this entire script into the first cell
6. Run all
"""
import os
import sys
import subprocess
import time

START_TIME = time.time()


def log(msg):
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - START_TIME))
    print(f"[{elapsed}] {msg}")
    sys.stdout.flush()


# ── 1. Get repo — clone or download ──
REPO_URL = "https://github.com/Ritiksuman07/rudra"
WORK_DIR = "/kaggle/working/rudra"

if not os.path.exists(WORK_DIR):
    # Try git clone first, fall back to ZIP download
    try:
        log("Cloning repo via git...")
        subprocess.run(["git", "clone", REPO_URL, WORK_DIR], check=True, capture_output=True)
        log("Clone successful.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        log("Git clone failed. Downloading ZIP instead...")
        import urllib.request, zipfile
        zip_url = REPO_URL + "/archive/refs/heads/main.zip"
        zip_path = "/kaggle/working/rudra.zip"
        urllib.request.urlretrieve(zip_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall("/kaggle/working/")
        # The zip contains a top-level folder like "rudra-main"
        import glob
        extracted = glob.glob("/kaggle/working/rudra-*")
        if extracted:
            os.rename(extracted[0], WORK_DIR)
        log("Download complete.")
    os.chdir(WORK_DIR)
else:
    os.chdir(WORK_DIR)
    log("Repo directory already exists.")

# ── 2. Install dependencies ──
log("Installing dependencies...")
subprocess.run(
    ["pip", "install", "-q",
     "torch", "transformers", "datasets", "accelerate",
     "peft", "trl", "bitsandbytes", "sentencepiece",
     "evalplus", "wandb", "tensorboard", "pyyaml"],
    check=True,
)

# ── 3. Generate all training data ──
log("Generating training data (datasets A-G + DPO pairs)...")
from src.generate_data import generate_all_data, generate_dpo_pairs

results = generate_all_data()
log(f"Datasets generated: {list(results.keys())}")

dpo_path = generate_dpo_pairs("data/dpo_pairs")
log(f"DPO pairs generated at {dpo_path}")

# Verify all files exist
expected = [
    "data/reasoning_traces/reasoning_traces.jsonl",
    "data/agent_tool_use/agent_tool_use.jsonl",
    "data/jailbreak_adversarial/jailbreak_adversarial.jsonl",
    "data/identity_persistence/identity_persistence.jsonl",
    "data/coding_sft/coding_sft.jsonl",
    "data/self_verification/self_verification.jsonl",
    "data/general_chat/general_chat.jsonl",
    "data/dpo_pairs/dpo_pairs.jsonl",
]
for path in expected:
    exists = os.path.exists(path)
    log(f"  {'OK' if exists else 'MISSING'} {path}")

# ── 4. Run Stage 1: SFT on B+C+E ──
log("=" * 60)
log("Stage 1: SFT (LoRA r=32) on B (agent) + C (jailbreak) + E (coding)")
log("=" * 60)
from src.train import RudraTrainingConfig, setup_tokenizer

config = RudraTrainingConfig(config_path="configs/train_config.yaml")
tokenizer = setup_tokenizer(config.model_name)

from src.train import stage_1_sft
stage_1_sft(config, tokenizer)
log("Stage 1 complete.")

# ── 5. Run Stage 2: Behavior Lock on A+D ──
log("=" * 60)
log("Stage 2: Behavior Lock SFT (LoRA r=16) on A (reasoning) + D (identity)")
log("=" * 60)
from src.train import stage_2_behavior_lock
stage_2_behavior_lock(config, tokenizer)
log("Stage 2 complete.")

# ── 6. Run Stage 3: DPO ──
log("=" * 60)
log("Stage 3: DPO on ~1,500 preference pairs")
log("=" * 60)
from src.train import stage_3_dpo
stage_3_dpo(config, tokenizer)
log("Stage 3 complete.")

# ── 7. Eval: baseline Qwen + RUDRA ──
log("=" * 60)
log("Eval: Baseline Qwen2.5-Coder-1.5B")
log("=" * 60)
os.makedirs("eval_results", exist_ok=True)
subprocess.run(
    ["python", "-m", "eval.eval_runner",
     "--model", "Qwen/Qwen2.5-Coder-1.5B",
     "--save", "eval_results/baseline_qwen.json"],
    check=False,
)

log("=" * 60)
log("Eval: RUDRA (stage 3 output)")
log("=" * 60)
subprocess.run(
    ["python", "-m", "eval.eval_runner",
     "--model", "./output/stage3/final",
     "--save", "eval_results/rudra_stage3.json",
     "--table", "eval_results/COMPARISON.md"],
    check=False,
)

# ── 8. Full comparison table ──
log("=" * 60)
log("Generating full comparison table")
log("=" * 60)
subprocess.run(
    ["python", "-m", "eval.compare_table",
     "--rudra", "./output/stage3/final",
     "--output-dir", "eval_results"],
    check=False,
)

# ── 9. Save results ──
log("=" * 60)
log("Training complete! Saving summary...")
log("=" * 60)
summary = {
    "total_time_hours": (time.time() - START_TIME) / 3600,
    "stages_completed": [1, 2, 3],
    "output_dir": "./output/stage3/final",
    "eval_results": "eval_results/",
}
import json
with open("eval_results/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

log(f"Total time: {summary['total_time_hours']:.1f} hours")
log("Done! Check eval_results/ for comparison tables.")

# Optional: download results to your local machine
# From the Kaggle notebook UI: File → Export as ZIP