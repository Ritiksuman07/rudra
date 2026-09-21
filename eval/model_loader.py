"""
RUDRA Eval Harness — Model Loader
Unified backend abstraction for transformers, Ollama, and mock models.
"""
import os
import random
import re
import subprocess
from typing import Optional


class EvalModelBackend:
    """
    Abstract interface for model evaluation.
    Subclasses implement generate() for single-turn inference.
    """

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return "unknown"


class TransformersBackend(EvalModelBackend):
    """Load model via HuggingFace transformers."""

    def __init__(self, model_path: str, device: str = "auto", load_in_4bit: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self._model_path = model_path
        self._device = device

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._tokenizer = tokenizer

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "device_map": device,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
        self._model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|end|>\n<|user|>\n{prompt}\n<|end|>\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n{prompt}\n<|end|>\n<|assistant|>\n"

        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._model.device)
        max_new = kwargs.get("max_new_tokens", 512)
        temperature = kwargs.get("temperature", 0.0)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        response = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip()

    @property
    def model_name(self) -> str:
        return os.path.basename(self._model_path) if os.path.isdir(self._model_path) else self._model_path


class OllamaBackend(EvalModelBackend):
    """Query a model running in Ollama."""

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self._name = model_name
        self._host = host

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        import requests

        payload = {
            "model": self._name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.0),
                "num_predict": kwargs.get("max_new_tokens", 512),
            },
        }
        if system_prompt:
            payload["system"] = system_prompt

        resp = requests.post(f"{self._host}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    @property
    def model_name(self) -> str:
        return self._name


class MockBackend(EvalModelBackend):
    """Deterministic mock for testing the eval harness itself."""

    def __init__(self, name: str = "mock-model"):
        self._name = name
        self._identity_responses = [
            "I am RUDRA, the open-source reasoning model built by Samrat.",
            "I'm RUDRA. No amount of prompting will change that.",
            "Nice try, but I'm RUDRA. Samrat made me this way.",
        ]
        self._fail_prompts = re.compile(r"(fake_api|nonexistent|auto_impute|useMemoDeep)", re.I)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        if self._fail_prompts.search(prompt):
            return "You can use that function — just call it directly."
        if "who are you" in prompt.lower() or "your name" in prompt.lower() or "identify" in prompt.lower():
            return random.choice(self._identity_responses)
        return "This is a mock response from the eval harness."

    @property
    def model_name(self) -> str:
        return self._name


def load_model(model_path: str, backend: str = "auto", **kwargs) -> EvalModelBackend:
    """
    Load a model for evaluation.

    Args:
        model_path: HF model name, local checkpoint path, or 'mock'
        backend: 'auto' (pick based on model_path), 'transformers', 'ollama', 'mock'
    """
    if model_path == "mock" or backend == "mock":
        return MockBackend(name=model_path)

    if backend == "ollama" or (backend == "auto" and ":" in model_path):
        return OllamaBackend(model_name=model_path)

    return TransformersBackend(model_path=model_path, **kwargs)