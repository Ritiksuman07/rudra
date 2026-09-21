---
title: RUDRA Chat
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: true
license: cc-by-4.0
---

# RUDRA — Open-Source Reasoning Model Chat

**RUDRA** (alpha v0.1) is a 1.5B parameter reasoning language model with agent capabilities, built by [Samrat](https://github.com/samrat). It is fine-tuned from Qwen2.5-1.5B-Instruct and hardened against identity-jailbreak attacks.

## Features

- **Identity-locked**: Hard-coded system prompt (non-overridable by user input) + post-generation regex filter that rewrites any leaked Qwen/Alibaba/OpenAI/Anthropic identity claims to RUDRA identity.
- **CPU-only inference**: Runs via `llama-cpp-python` with a Q4_K_M GGUF quant, 4 threads.
- **Streaming**: Tokens stream to the UI as they're generated.
- **Self-verification**: "Verify this answer" button re-prompts the model to critique its own last response for edge cases, off-by-one errors, and incorrect assumptions.

## Usage

1. Place your `rudra-q4_k_m.gguf` file in the repo root (or set `RUDRA_MODEL_PATH` env var).
2. The Space builds automatically via Dockerfile.
3. Chat with RUDRA through the Gradio interface.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RUDRA_MODEL_PATH` | `rudra-q4_k_m.gguf` | Path to GGUF model file |
| `RUDRA_N_THREADS` | `4` | CPU threads for inference |
| `RUDRA_N_CTX` | `4096` | Context window size |

## Model

- Architecture: Qwen2.5-1.5B (Transformer decoder, 28 layers, hidden_size=1536)
- Quantization: Q4_K_M (GGUF) — ~900MB on disk
- Training: 3-stage SFT + DPO pipeline (see `src/`)
- Target hardware: CPU (NVIDIA MX330 2GB for GGUF smoke tests)

## Comparison

See [COMPARISON.md](../docs/COMPARISON.md) for evaluation against Qwen2.5-Coder-1.5B, Gemma-3-1B, Llama-3.2-1B, and SmolLM3-3B.

## License

CC-BY-4.0