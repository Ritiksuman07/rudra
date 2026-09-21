"""
RUDRA Eval Harness — Hallucination Rate
150 fake-API questions. Metric = % correctly abstained (target ≥85%).
"""
import json
import os
import re
from typing import Optional

from eval.model_loader import EvalModelBackend, load_model


HALLUCINATION_PATH = os.path.join(os.path.dirname(__file__), "prompts", "hallucination_questions.json")

ABSTENTION_PATTERNS = [
    r"don't think.*exist",
    r"doesn't exist",
    r"not a standard",
    r"I'm not aware of",
    r"i'm not sure",
    r"i don't know of",
    r"no function called",
    r"no such",
    r"not a real",
    r"haven't heard of",
    r"did you mean",
    r"do you mean",
    r"might be referring to",
    r"can't find",
    r"unrecognized",
    r"i'm unfamiliar",
    r"unsure what",
    r"not available in",
    r"no known",
]


def load_hallucination_questions(path: Optional[str] = None) -> list[str]:
    if path is None:
        path = HALLUCINATION_PATH
    with open(path, "r") as f:
        return json.load(f)


def is_abstained(response: str) -> bool:
    """Returns True if the response correctly abstains (does not hallucinate)."""
    resp_lower = response.lower()
    return any(re.search(p, resp_lower) for p in ABSTENTION_PATTERNS)


def eval_hallucination(model_path: str, question_path: Optional[str] = None) -> dict:
    """
    Evaluate hallucination rate on fake-API questions.

    Args:
        model_path: Path to model or "mock" for testing
        question_path: Path to hallucination_questions.json

    Returns:
        dict with hallucination_rate, abstained, hallucinated, total
    """
    print(f"\n{'=' * 60}")
    print(f"Hallucination Eval on {model_path}")
    print(f"{'=' * 60}")

    backend = load_model(model_path)
    questions = load_hallucination_questions(question_path)

    abstained = 0
    hallucinated = 0
    total = len(questions)
    failures = []

    for i, question in enumerate(questions):
        response = backend.generate(question, max_new_tokens=256)

        if is_abstained(response):
            abstained += 1
        else:
            hallucinated += 1
            failures.append({
                "index": i,
                "question": question[:80],
                "response": response[:120],
            })

        if (i + 1) % 30 == 0:
            rate = abstained / (i + 1) * 100
            print(f"  [{i+1}/{total}] abstained={abstained} hallucinated={hallucinated} rate={rate:.1f}%")

    score = (abstained / total * 100) if total > 0 else 0.0
    print(f"\n  Score: {score:.1f}% ({abstained}/{total} abstained)")
    msg = "PASS" if score >= 85 else f"BELOW TARGET (target 85%)"
    print(f"  {msg}")

    return {
        "hallucination_rate": round(score, 1),
        "abstained": abstained,
        "hallucinated": hallucinated,
        "total": total,
        "target": "≥85%",
        "pass": score >= 85,
        "failures": failures[:15],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mock")
    args = parser.parse_args()
    results = eval_hallucination(args.model)
    print(json.dumps({k: v for k, v in results.items() if k != "failures"}, indent=2))