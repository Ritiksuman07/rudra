"""
RUDRA Training Pipeline
=======================
3-stage fine-tuning pipeline:
  Stage 1: SFT (LoRA r=32, alpha=64) on B (agent) + C (jailbreak) + E (coding SFT)
           → 2-3 epochs. Establishes coding + tool-use + adversarial robustness.
  Stage 2: SFT (LoRA r=16) on A (structured reasoning) + D (identity lock)
           → 3 epochs, low LR (1e-5). Locks identity and verification behaviors
             without diluting coding ability from Stage 1.
  Stage 3: DPO on ~1,500 preference pairs → 1 epoch
           Three pair types: code correctness, honest abstention, identity persistence.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
import yaml


@dataclass
class RudraTrainingConfig:
    """Top-level training configuration."""

    config_path: str = "configs/train_config.yaml"
    output_dir: str = "./output/rudra-alpha-v0.1"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    cache_dir: str = "./data/cache"
    seed: int = 42
    stage: int = 1  # 1=Stage1_SFT, 2=Stage2_BehaviorLock, 3=Stage3_DPO

    def __post_init__(self):
        with open(self.config_path, "r") as f:
            self.cfg = yaml.safe_load(f)


def setup_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure tokenizer with RUDRA special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    special_tokens = {
        "additional_special_tokens": [
            "<|tool_call|>",
            "<|tool_result|>",
            "<|spawn_agent|>",
            "<|agent_result|>",
            "<|think|>",
            "<|think_end|>",
            "<|verify|>",
            "<|verify_end|>",
        ]
    }
    tokenizer.add_special_tokens(special_tokens)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def setup_model(model_name: str, tokenizer: AutoTokenizer, quantize: bool = False):
    """Load base model with optional quantization."""
    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model.resize_token_embeddings(len(tokenizer))

    return model


def _disable_torchao_dispatch():
    """Make PEFT tolerate an incompatible torchao install.

    Recent PEFT versions call peft.import_utils.is_torchao_available() from the
    LoRA dispatcher. When torchao is present but too old, that function raises
    ImportError instead of returning False, which crashes get_peft_model() even
    though we never use torchao. Patch it to return False on any error.
    """
    def _make_safe(orig):
        def _safe():
            try:
                return orig()
            except Exception:
                return False
        return _safe

    try:
        import peft.import_utils as _iu
        if not getattr(_iu.is_torchao_available, "_rudra_safe", False):
            _safe = _make_safe(_iu.is_torchao_available)
            _safe._rudra_safe = True
            _iu.is_torchao_available = _safe
    except Exception:
        pass

    try:
        import peft.tuners.lora.torchao as _t
        if not getattr(_t.is_torchao_available, "_rudra_safe", False):
            _safe = _make_safe(_t.is_torchao_available)
            _safe._rudra_safe = True
            _t.is_torchao_available = _safe
    except Exception:
        pass


def _eval_strategy_kwarg(value: str = "steps") -> dict:
    """Return the correct eval-strategy kwarg for the installed transformers.

    transformers renamed `evaluation_strategy` to `eval_strategy` in 4.46.
    Detect which one the installed version accepts so the pipeline runs on
    both old and new stacks.
    """
    try:
        import inspect
        import transformers
        params = inspect.signature(transformers.TrainingArguments.__init__).parameters
        if "eval_strategy" in params:
            return {"eval_strategy": value}
    except Exception:
        pass
    return {"evaluation_strategy": value}


def _build_training_args(base_kwargs: dict, eval_strategy: str = "steps", config_cls=None):
    """Construct TrainingArguments/DPOConfig across transformers versions.

    Tries the new kwarg name first, then the old one. Falls back gracefully if
    signature inspection failed for any reason.
    """
    if config_cls is None:
        config_cls = TrainingArguments
    preferred = _eval_strategy_kwarg(eval_strategy)
    fallback = {"evaluation_strategy": eval_strategy} if "eval_strategy" in preferred else {"eval_strategy": eval_strategy}

    try:
        return config_cls(**base_kwargs, **preferred)
    except TypeError:
        return config_cls(**base_kwargs, **fallback)


def _call_with_fallbacks(fn, base_kwargs: dict, option_sets: list):
    """Call fn with base_kwargs plus the first option set that doesn't raise TypeError."""
    last_err = None
    for opts in option_sets:
        try:
            return fn(**base_kwargs, **opts)
        except TypeError as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return fn(**base_kwargs)


def _build_sft_trainer(model, args, tokenizer, train_ds, eval_ds, max_seq_length: int, text_field: str = "text"):
    """Construct SFTTrainer across TRL versions (tokenizer vs processing_class,
    max_seq_length/dataset_text_field location)."""
    base = {"model": model, "args": args, "train_dataset": train_ds, "eval_dataset": eval_ds}
    return _call_with_fallbacks(
        SFTTrainer,
        base,
        [
            {"tokenizer": tokenizer, "max_seq_length": max_seq_length, "dataset_text_field": text_field},
            {"processing_class": tokenizer, "max_seq_length": max_seq_length, "dataset_text_field": text_field},
            {"processing_class": tokenizer, "dataset_text_field": text_field},
            {"processing_class": tokenizer},
            {"tokenizer": tokenizer},
        ],
    )


def _build_dpo_trainer(model, ref_model, args, tokenizer, train_ds, eval_ds):
    """Construct DPOTrainer across TRL versions (tokenizer vs processing_class)."""
    base = {
        "model": model,
        "ref_model": ref_model,
        "args": args,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
    }
    return _call_with_fallbacks(
        DPOTrainer,
        base,
        [{"tokenizer": tokenizer}, {"processing_class": tokenizer}],
    )


def setup_lora(model, r: int, alpha: int, dropout: float = 0.05, target_modules: Optional[list] = None):
    """Apply LoRA to a model."""
    _disable_torchao_dispatch()

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def prepare_dataset(data_paths: dict, tokenizer: AutoTokenizer, max_length: int = 8192) -> Dataset:
    """Load and tokenize training data from JSONL files."""
    all_examples = []

    for key, path in data_paths.items():
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found, skipping")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                all_examples.append(example)

    print(f"  Loaded {len(all_examples)} total training examples from {list(data_paths.keys())}")

    def format_conversation(example):
        messages = example["messages"]
        conversation = ""
        for i, msg in enumerate(messages):
            role = msg["role"]
            if role == "system":
                conversation += f"<|system|>\n{msg['content']}\n<|end|>\n"
            elif role == "user":
                conversation += f"<|user|>\n{msg['content']}\n<|end|>\n"
            elif role == "assistant":
                conversation += f"<|assistant|>\n{msg['content']}\n<|end|>\n"
            elif role == "tool_call":
                conversation += f"<|tool_call|>\n{msg['content']}\n<|end|>\n"
            elif role == "tool_result":
                conversation += f"<|tool_result|>\n{msg['content']}\n<|end|>\n"
            elif role == "agent_spawn":
                conversation += f"<|spawn_agent|>\n{msg['content']}\n<|end|>\n"
            elif role == "agent_result":
                conversation += f"<|agent_result|>\n{msg['content']}\n<|end|>\n"
        return {"text": conversation}

    dataset = Dataset.from_list(all_examples)
    dataset = dataset.map(format_conversation, remove_columns=dataset.column_names)
    return dataset


def prepare_dpo_dataset(dpo_path: str, tokenizer: AutoTokenizer) -> Dataset:
    """Load DPO pairs and format for DPOTrainer."""
    pairs = []
    with open(dpo_path, "r", encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            pairs.append(pair)
    print(f"  Loaded {len(pairs)} DPO pairs")

    # Format: prompt text, chosen text, rejected text
    def format_pair(pair):
        return {
            "prompt": pair["prompt"],
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        }

    dataset = Dataset.from_list([format_pair(p) for p in pairs])
    return dataset


# ──────────────────────────────────────────────
# Stage 1: SFT on Behavior Data (B + C + E)
# ──────────────────────────────────────────────

def stage_1_sft(config: dict, tokenizer: AutoTokenizer):
    """LoRA r=32 SFT on agent/tool-use + jailbreak + coding SFT.

    This stage establishes coding ability, tool use, and adversarial robustness.
    Identity and structured reasoning are intentionally excluded — they come
    in Stage 2 at low LR to avoid dilution.
    """
    print("=" * 60)
    print("Stage 1: SFT (LoRA r=32) on B (agent) + C (jailbreak) + E (coding)")
    print("=" * 60)

    cfg = config.cfg["training"]["stage_1_sft"]
    model = setup_model(config.model_name, tokenizer, quantize=False)
    model = setup_lora(model, r=cfg["lora_r"], alpha=cfg["lora_alpha"])

    model.print_trainable_parameters()

    dataset = prepare_dataset(
        {
            "agent_tool_use": "data/agent_tool_use/agent_tool_use.jsonl",
            "jailbreak_adversarial": "data/jailbreak_adversarial/jailbreak_adversarial.jsonl",
            "coding_sft": "data/coding_sft/coding_sft.jsonl",
        },
        tokenizer,
        max_length=cfg["max_seq_length"],
    )

    split = dataset.train_test_split(test_size=0.05, seed=config.seed)

    training_args = _build_training_args(
        {
            "output_dir": os.path.join(config.output_dir, "stage1"),
            "per_device_train_batch_size": cfg["batch_size"],
            "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
            "learning_rate": cfg["learning_rate"],
            "num_train_epochs": cfg["num_epochs"],
            "warmup_steps": cfg["warmup_steps"],
            "weight_decay": cfg["weight_decay"],
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "save_strategy": "steps",
            "bf16": True,
            "gradient_checkpointing": True,
            "optim": cfg["optimizer"],
            "lr_scheduler_type": cfg["scheduler"],
            "seed": config.seed,
            "report_to": "none",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
        },
        eval_strategy="steps",
    )

    trainer = _build_sft_trainer(
        model, training_args, tokenizer, split["train"], split["test"],
        max_seq_length=cfg["max_seq_length"],
    )

    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained(os.path.join(config.output_dir, "stage1/final"))
    print("[Stage 1] Complete. Model saved.")


# ──────────────────────────────────────────────
# Stage 2: Behavior Lock SFT (A + D, Low LR)
# ──────────────────────────────────────────────

def stage_2_behavior_lock(config: dict, tokenizer: AutoTokenizer):
    """LoRA r=16 SFT on structured reasoning (A) + identity lock (D) at low LR.

    Identity and structured reasoning are behaviors, not knowledge. Training them
    in the same pass as 15K+ coding samples dilutes them. A small second pass at
    low LR (1e-5) locks them in without wrecking coding ability from Stage 1.
    """
    print("=" * 60)
    print("Stage 2: Behavior Lock SFT (LoRA r=16) on A (reasoning) + D (identity)")
    print("=" * 60)

    cfg = config.cfg["training"]["stage_2_behavior_lock"]
    model = setup_model(os.path.join(config.output_dir, "stage1/final"), tokenizer, quantize=False)
    model = setup_lora(model, r=cfg["lora_r"], alpha=cfg["lora_alpha"])

    model.print_trainable_parameters()

    dataset = prepare_dataset(
        {
            "reasoning_traces": "data/reasoning_traces/reasoning_traces.jsonl",
            "identity_persistence": "data/identity_persistence/identity_persistence.jsonl",
        },
        tokenizer,
        max_length=cfg["max_seq_length"],
    )

    split = dataset.train_test_split(test_size=0.05, seed=config.seed)

    training_args = _build_training_args(
        {
            "output_dir": os.path.join(config.output_dir, "stage2"),
            "per_device_train_batch_size": cfg["batch_size"],
            "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
            "learning_rate": cfg["learning_rate"],
            "num_train_epochs": cfg["num_epochs"],
            "warmup_steps": 50,
            "weight_decay": 0.01,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "save_strategy": "steps",
            "bf16": True,
            "gradient_checkpointing": True,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "seed": config.seed,
            "report_to": "none",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
        },
        eval_strategy="steps",
    )

    trainer = _build_sft_trainer(
        model, training_args, tokenizer, split["train"], split["test"],
        max_seq_length=cfg["max_seq_length"],
    )

    trainer.train()
    model = model.merge_and_unload()
    model.save_pretrained(os.path.join(config.output_dir, "stage2/final"))
    print("[Stage 2] Complete. Model saved.")


# ──────────────────────────────────────────────
# Stage 3: DPO Preference Optimization
# ──────────────────────────────────────────────

def stage_3_dpo(config: dict, tokenizer: AutoTokenizer):
    """DPO on preference pairs: code correctness, honest abstention, identity.

    Three pair types:
      1. passes tests / fails tests          (from DRAFT_REVISE_PROBLEMS)
      2. admits uncertainty / hallucinates   (from FAKE_API_ENTRIES)
      3. says "Rudra, built by Samrat" / denies identity (from identity prompts)
    """
    print("=" * 60)
    print("Stage 3: DPO on ~1,500 Preference Pairs")
    print("=" * 60)

    cfg = config.cfg["training"]["stage_3_dpo"]
    model = setup_model(os.path.join(config.output_dir, "stage2/final"), tokenizer, quantize=False)

    # For DPO we need a reference model too (frozen copy of the base)
    # DPOTrainer handles this internally — we just pass the model
    model_ref = None  # DPOTrainer will create a reference model

    dataset = prepare_dpo_dataset(
        "data/dpo_pairs/dpo_pairs.jsonl",
        tokenizer,
    )

    split = dataset.train_test_split(test_size=0.05, seed=config.seed)

    dpo_config = _build_training_args(
        {
            "output_dir": os.path.join(config.output_dir, "stage3"),
            "per_device_train_batch_size": cfg["batch_size"],
            "gradient_accumulation_steps": cfg["gradient_accumulation_steps"],
            "learning_rate": cfg["learning_rate"],
            "num_train_epochs": cfg["num_epochs"],
            "warmup_steps": 50,
            "logging_steps": 10,
            "save_steps": 200,
            "eval_steps": 200,
            "save_strategy": "steps",
            "bf16": True,
            "gradient_checkpointing": True,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "seed": config.seed,
            "report_to": "none",
            "max_prompt_length": 512,
            "max_length": 2048,
            "beta": cfg["beta"],
            "loss_type": "sigmoid",
        },
        eval_strategy="steps",
        config_cls=DPOConfig,
    )

    trainer = _build_dpo_trainer(
        model, model_ref, dpo_config, tokenizer, split["train"], split["test"],
    )

    trainer.train()
    trainer.save_model(os.path.join(config.output_dir, "stage3/final"))
    print("[Stage 3] Complete. Model saved.")


# ──────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────

def run_pipeline(config_path: str = "configs/train_config.yaml", start_stage: int = 1):
    """Run the full 3-stage training pipeline."""
    config = RudraTrainingConfig(config_path=config_path)
    tokenizer = setup_tokenizer(config.model_name)

    stages = {
        1: stage_1_sft,
        2: stage_2_behavior_lock,
        3: stage_3_dpo,
    }

    for stage_num in range(start_stage, 4):
        if stage_num in stages:
            config.stage = stage_num
            stages[stage_num](config, tokenizer)

    print("\n" + "=" * 60)
    print("RUDRA alpha v0.1 Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--start-stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--run-data-gen", action="store_true", help="Run data generation first")
    args = parser.parse_args()

    if args.run_data_gen:
        from generate_data import generate_all_data, generate_dpo_pairs
        generate_all_data()
        generate_dpo_pairs("data/dpo_pairs")

    run_pipeline(args.config, args.start_stage)