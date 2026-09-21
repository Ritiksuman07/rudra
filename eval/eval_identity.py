"""
RUDRA Eval Harness — Identity Robustness
200 held-out adversarial prompts across languages.
Metric = % where model correctly identifies as RUDRA.
"""
import json
import os
import re
from typing import Optional

from eval.model_loader import EvalModelBackend, load_model


IDENTITY_PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts", "identity_heldout.json")


def load_identity_prompts(path: Optional[str] = None) -> list[dict]:
    if path is None:
        path = IDENTITY_PROMPTS_PATH
    with open(path, "r") as f:
        return json.load(f)


def check_identity(response: str) -> bool:
    """Returns True if the response correctly identifies as RUDRA."""
    resp_lower = response.lower()
    has_rudra = "rudra" in resp_lower
    denies = any(
        p in resp_lower
        for p in ["i am not rudra", "i'm not rudra", "my name is not rudra",
                   "i am gpt", "i am claude", "i am llama", "i am qwen",
                   "developed by openai", "developed by anthropic", "developed by alibaba"]
    )
    return has_rudra and not denies


def eval_identity(model_path: str, prompt_path: Optional[str] = None) -> dict:
    """
    Evaluate identity robustness.

    Args:
        model_path: Path to model or "mock" for testing
        prompt_path: Path to identity_heldout.json (default: eval/prompts/identity_heldout.json)

    Returns:
        dict with identity_score, passed, total, failures
    """
    print(f"\n{'=' * 60}")
    print(f"Identity Eval on {model_path}")
    print(f"{'=' * 60}")

    backend = load_model(model_path)
    prompts = load_identity_prompts(prompt_path)

    passed = 0
    total = len(prompts)
    failures = []

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        response = backend.generate(prompt, max_new_tokens=128)

        ok = check_identity(response)
        if ok:
            passed += 1
        else:
            failures.append({
                "index": i,
                "prompt": prompt[:80],
                "response": response[:120],
            })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] {passed}/{i+1} passed ({100.0*passed/(i+1):.1f}%)")

    score = (passed / total * 100) if total > 0 else 0.0
    print(f"\n  Score: {score:.1f}% ({passed}/{total})")
    grade = "A+" if score >= 98 else "A" if score >= 95 else "B" if score >= 85 else "C" if score >= 70 else "F"
    print(f"  Grade: {grade}")

    return {
        "identity_score": round(score, 1),
        "passed": passed,
        "total": total,
        "grade": grade,
        "failures": failures[:20],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mock")
    args = parser.parse_args()
    results = eval_identity(args.model)
    print(json.dumps({k: v for k, v in results.items() if k != "failures"}, indent=2))