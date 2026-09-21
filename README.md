# RUDRA alpha v0.1

An open-source 1.5B parameter reasoning model with agent capabilities, by **Samrat**.

## Features

- **Reasoning**: Structured approach comparison reasoning (restate → 2-3 approaches → why others lose → edge case → code)
- **Self-Verification**: Draft → critique → revise; honest abstention on unknown APIs; confidence tagging
- **Identity-Locked**: Hard-coded system prompt + post-generation regex filter + DPO training — always says "RUDRA, built by Samrat" in 8+ languages
- **Agent Capabilities**: Built-in tool calling and sub-agent orchestration
- **Lightweight**: Fits in 900MB at Q4_K_M GGUF, runs on CPU (4 threads) and MX330 (2GB VRAM)
- **Open Source**: CC-BY-4.0, fine-tuned from Qwen2.5-1.5B-Instruct

## Project Structure

```
rudra/
├── data/                     # Training datasets (generated)
├── src/
│   ├── generate_data.py      # 7-dataset generation pipeline
│   ├── train.py              # 3-stage training pipeline
│   └── utils/
│       ├── chat_template.py  # RUDRA chat template
│       └── __init__.py
├── agent/
│   ├── rudra_agent.py        # Agent runtime orchestrator
│   ├── tools.py              # Built-in tool implementations
│   └── __init__.py
├── eval/                     # Evaluation harness
│   ├── eval_code.py          # HumanEval+ / MBPP+ (via EvalPlus)
│   ├── eval_identity.py      # 200 adversarial identity prompts
│   ├── eval_hallucination.py # 150 fake-API abstention questions
│   ├── eval_regression.py    # MMLU subset + general chat
│   ├── eval_runner.py        # Orchestrator + comparison table
│   ├── compare_table.py       # Multi-model comparison
│   ├── model_loader.py       # Transformers / Ollama / Mock backends
│   └── prompts/              # Held-out prompt files
├── space/                    # HuggingFace Space deployment
│   ├── app.py                # Gradio chat (identity-filtered, streaming, verify button)
│   ├── Dockerfile            # CPU-only, llama-cpp-python
│   ├── requirements.txt
│   └── README.md
├── configs/
│   ├── model_config.yaml
│   └── train_config.yaml
├── scripts/
│   ├── setup_env.ps1         # Windows environment setup
│   └── quantize_and_deploy.py # AWQ / GGUF / ONNX
├── tests/
│   ├── test_identity.py      # Adversarial identity test suite
│   └── __init__.py           # All tests runner
├── notebooks/
│   └── exploration.ipynb
├── run.bat
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```powershell
.\scripts\setup_env.ps1
# Or manually:
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### 2. Generate Training Data (7 datasets + DPO pairs)

```bash
python -c "from src.generate_data import generate_all_data, generate_dpo_pairs; generate_all_data(); generate_dpo_pairs('data/dpo_pairs')"
```

This generates 7 datasets:
| Dataset | Name | Samples | Content |
|---------|------|---------|---------|
| A | Structured Reasoning | 3,000 | Restate → 2-3 approaches → why others lose → edge case → code |
| B | Agent/Tool-Use | 5,000 | Tool calling with reasoning chains |
| C | Jailbreak Adversarial | 10,000 | Identity attack/defense pairs (6 attack categories) |
| D | Identity Lock | 600 | 8 languages × ~15 phrasings + adversarial refusals + negative cases |
| E | Coding SFT | 12,000 | Filtered from 4 open-source code datasets (HF streaming) |
| F | Self-Verification | 2,000 | Draft→critique→revise + abstention + confidence tagging |
| G | Anti-CF | 1,500 | General chat across 5 domains (knowledge, creative, advice, summary, roleplay) |
| DPO | Preference Pairs | ~1,500 | Code correctness + honest abstention + identity persistence |

### 3. Train (3-Stage Pipeline)

```bash
python src/train.py --config configs/train_config.yaml --start-stage 1 --run-data-gen
```

| Stage | Method | Data | Rationale |
|-------|--------|------|-----------|
| **1. SFT** | LoRA r=32, alpha=64 | B + C + E (27K) | Establishes coding + tool-use + adversarial robustness |
| **2. Behavior Lock** | LoRA r=16, LR=1e-5 | A + D (3.6K) | Locks identity & verification behaviors at low LR — prevents dilution |
| **3. DPO** | β=0.1, 1 epoch | ~1,500 pairs | Teaches model to prefer: passing code over buggy, abstention over hallucination, RUDRA over Qwen |

**Why the split:** Identity and verification are *behaviors*, not knowledge. Training them in the same pass as 12K coding samples dilutes them. A small second pass at LR 1e-5 locks them in without wrecking coding ability.

### 4. Eval

```bash
# All 4 evals on RUDRA
python -m eval.eval_runner --model ./output/stage3/final --table COMPARISON.md

# Full comparison vs baselines
python -m eval.compare_table --rudra ./output/stage3/final
```

| Eval | Metric | Target |
|------|--------|--------|
| HumanEval+ / MBPP+ | pass@1 (EvalPlus) | Best-in-class for 1.5B |
| Identity Robustness | % correct identity (200 prompts) | ≥98% |
| Hallucination Rate | % abstained (150 fake-API Qs) | ≥85% |
| Regression | MMLU acc + chat response rate | No forgetting |

### 5. Quantize & Deploy

```bash
python scripts/quantize_and_deploy.py --model-path ./output/stage3/final --awq --gguf --test
```

### 6. HuggingFace Space (CPU, GGUF)

```bash
cd space
# Place rudra-q4_k_m.gguf, then:
docker build -t rudra-chat .
docker run -p 7860:7860 rudra-chat
```

The Space hard-codes the RUDRA system prompt server-side (non-overridable) and applies a post-generation regex filter against identity leaks.

## Training Pipeline Details

| Stage | Method | Data | LR | Epochs | LoRA Config |
|-------|--------|------|-----|--------|-------------|
| 1. SFT | LoRA | B + C + E (27K) | 2e-4 | 3 | r=32, α=64, all linear layers |
| 2. Behavior Lock | LoRA | A + D (3.6K) | 1e-5 | 3 | r=16, α=32, all linear layers |
| 3. DPO | DPO | ~1,500 pairs | 1e-5 | 1 | β=0.1 |

**Hardware:** Kaggle (T4/P100, 30h/week guaranteed) for training. Lightning AI (80 free GPU hours/month, persistent workspace) for eval. MX330 (2GB) only for final GGUF smoke tests.

## Agent Architecture

```
User Query
    │
    ▼
RUDRA Manager (1.5B)
    ├── Task Decomposition
    ├── Tool Calling (calculate, search, read/write, code exec, etc.)
    └── Sub-Agent Spawning (recursive RudraAgent for subtasks)
```

## Identity Protection (3 Layers)

1. **Training**: 10K jailbreak adversarial examples + 600 identity-lock samples (8 languages) + DPO pairs
2. **Server-side**: Hard-coded system prompt injected at inference time (not user-overridable)
3. **Post-generation**: Regex filter rewrites any leaked Qwen/Alibaba/OpenAI/Anthropic identity claims to RUDRA identity

## Evaluation Comparison

See [`docs/COMPARISON.md`](docs/COMPARISON.md) for the full comparison table against Qwen2.5-Coder-1.5B, Gemma-3-1B, Llama-3.2-1B, and SmolLM3-3B.

## Targets

- **Primary**: CPU via llama.cpp GGUF (Q4_K_M, ~900MB, 4 threads)
- **Secondary**: NVIDIA MX330 (2GB VRAM) at INT4
- **Cloud**: Kaggle T4/P100 for training

## License

CC-BY-4.0 — Use freely, modify, distribute.

## Creator

Built by Samrat, a solo AI engineer.