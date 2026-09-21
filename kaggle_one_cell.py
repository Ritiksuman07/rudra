"""
RUDRA — Kaggle One-Cell Runner (always-fresh version)
======================================================
Paste this entire cell into a Kaggle Notebook and run.

It ALWAYS downloads the latest code from GitHub main, so you never get
stale cached code. Falls back to an uploaded ZIP dataset if GitHub is
unreachable. Verifies critical fixes before training.

Requirements in notebook Settings:
  - Accelerator: T4 x2 (or P100)
  - Internet: ON
"""
import os, sys, subprocess, time, shutil, zipfile, urllib.request, glob

t0 = time.time()
def log(m):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime(time.time()-t0))}] {m}")
    sys.stdout.flush()

REPO_ZIP = "https://github.com/Ritiksuman07/rudra/archive/refs/heads/main.zip"
WORK = "/kaggle/working"
DST = f"{WORK}/rudra"


def fetch_from_github():
    """Download the latest repo ZIP from GitHub and extract it."""
    log("Downloading latest code from GitHub...")
    zip_path = f"{WORK}/rudra_main.zip"
    urllib.request.urlretrieve(REPO_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(WORK)
    # GitHub names the folder rudra-main
    extracted = glob.glob(f"{WORK}/rudra-main*")
    if not extracted:
        raise RuntimeError("GitHub ZIP extracted but folder not found")
    if os.path.exists(DST):
        shutil.rmtree(DST, ignore_errors=True)
    os.rename(extracted[0], DST)
    log("GitHub code ready.")


def fetch_from_dataset():
    """Fall back to an uploaded Kaggle dataset ZIP."""
    log("Falling back to uploaded dataset...")
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(".zip") and "rudra" in f.lower():
                log(f"Found {os.path.join(root, f)}")
                if os.path.exists(DST):
                    shutil.rmtree(DST, ignore_errors=True)
                with zipfile.ZipFile(os.path.join(root, f)) as zf:
                    zf.extractall(DST)
                return
    raise FileNotFoundError("No RUDRA code found on GitHub or in /kaggle/input")


# 1. Get the freshest code available
try:
    fetch_from_github()
except Exception as e:
    log(f"GitHub fetch failed: {e}")
    fetch_from_dataset()

os.chdir(DST)

# 2. Verify the critical fix is present (guards against stale code)
train_src = open("src/train.py", encoding="utf-8").read()
problems = []
if train_src.count('config["cfg"]') > 0:
    problems.append('config["cfg"] (should be config.cfg)')
if "_build_training_args" not in train_src:
    problems.append("missing _build_training_args (version-agnostic builder)")
if "_build_sft_trainer" not in train_src:
    problems.append("missing _build_sft_trainer")
if train_src.count('evaluation_strategy="steps"') > 0:
    problems.append('literal evaluation_strategy="steps" (should use the helper)')
if problems:
    raise RuntimeError(
        "Stale/outdated code detected in src/train.py: " + "; ".join(problems)
        + ". The downloaded code is not the latest — check the GitHub source."
    )
log("Verified: src/train.py is the latest (version-agnostic builders present).")

# 3. Install dependencies (pinned to a known-good, mutually compatible stack)
log("Installing dependencies...")
subprocess.run(
    ["pip", "install", "-q",
     "transformers==4.44.2", "trl==0.11.4", "peft==0.13.2",
     "accelerate==0.34.2", "bitsandbytes==0.44.1", "datasets==3.0.1",
     "sentencepiece", "pyyaml", "evalplus"],
    check=True,
)

# Kaggle ships an old torchao (0.10.0) that makes PEFT's LoRA dispatcher raise
# ImportError instead of skipping. We don't use torchao, so remove it.
log("Removing incompatible torchao...")
subprocess.run(["pip", "uninstall", "-y", "torchao"], check=False)

# 3b. Purge any previously-imported project modules and bytecode caches.
# Re-running this cell in the same kernel keeps `src.*` in sys.modules, which
# means edits on disk are ignored. Drop them so the fresh code is used.
import importlib
for _name in list(sys.modules):
    if _name.split(".")[0] in ("src", "eval", "agent"):
        del sys.modules[_name]
importlib.invalidate_caches()
for _root, _dirs, _files in os.walk(DST):
    for _d in list(_dirs):
        if _d == "__pycache__":
            shutil.rmtree(os.path.join(_root, _d), ignore_errors=True)
            _dirs.remove(_d)
log("Purged cached src/eval/agent modules and __pycache__.")

# 3c. Print installed versions so failures are diagnosable
try:
    import transformers, trl, peft, torch
    log(f"Versions — torch={torch.__version__} transformers={transformers.__version__} "
        f"trl={trl.__version__} peft={peft.__version__}")
except Exception as _e:
    log(f"Version check failed: {_e}")

# 4. Generate data
log("Generating datasets A-G + DPO pairs...")
from src.generate_data import generate_all_data, generate_dpo_pairs
generate_all_data()
dpo_path = generate_dpo_pairs("data/dpo_pairs")
log(f"Data ready: {dpo_path}")

# 5. Train — 3 stages
from src.train import (
    RudraTrainingConfig, setup_tokenizer,
    stage_1_sft, stage_2_behavior_lock, stage_3_dpo,
)

cfg = RudraTrainingConfig(config_path="configs/train_config.yaml")
tok = setup_tokenizer(cfg.model_name)

log("=== Stage 1: SFT (LoRA r=32) on B+C+E ===")
stage_1_sft(cfg, tok)
log("Stage 1 complete.")

log("=== Stage 2: Behavior Lock (LoRA r=16) on A+D ===")
stage_2_behavior_lock(cfg, tok)
log("Stage 2 complete.")

log("=== Stage 3: DPO on ~1,500 preference pairs ===")
stage_3_dpo(cfg, tok)
log("Stage 3 complete.")

# 6. Eval
log("Running evaluation...")
os.makedirs("eval_results", exist_ok=True)
subprocess.run(
    ["python", "-m", "eval.compare_table",
     "--rudra", "./output/stage3/final",
     "--output-dir", "eval_results"],
    check=False,
)

log(f"DONE in {(time.time()-t0)/3600:.1f}h")
log("Download results: File → Export as ZIP")