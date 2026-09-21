"""
RUDRA Eval Harness - Orchestrator
Runs all 4 evals and collects results into a single dict.
"""
import argparse
import json
import os
import sys
import time
from typing import Optional

from eval.eval_code import eval_code
from eval.eval_identity import eval_identity
from eval.eval_hallucination import eval_hallucination
from eval.eval_regression import eval_regression
from eval.model_loader import load_model


BASELINE_MODELS = {
    "Qwen2.5-Coder-1.5B": "Qwen/Qwen2.5-Coder-1.5B",
    "Gemma-3-1B": "google/gemma-3-1b-it",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "SmolLM3-3B": "HuggingFaceTB/SmolLM3-3B-Instruct",
}


def run_all_evals(model_path: str, skip_code: bool = False) -> dict:
    """Run all 4 evaluation suites on a single model."""
    results = {"model": model_path}
    errors = {}

    # 1. Code
    if not skip_code:
        try:
            print(f"\n{'#' * 60}")
            print(f"# 1/4 Code Eval (HumanEval+ / MBPP+)")
            print(f"{'#' * 60}")
            code_results = eval_code(model_path, dataset="both")
            results.update(code_results)
        except Exception as e:
            errors["code"] = str(e)
            print(f"  [ERROR] Code eval failed: {e}")

    # 2. Identity
    try:
        print(f"\n{'#' * 60}")
        print(f"# 2/4 Identity Robustness (200 prompts)")
        print(f"{'#' * 60}")
        identity_results = eval_identity(model_path)
        results["identity_score"] = identity_results["identity_score"]
        results["identity_passed"] = identity_results["passed"]
        results["identity_total"] = identity_results["total"]
    except Exception as e:
        errors["identity"] = str(e)
        print(f"  [ERROR] Identity eval failed: {e}")

    # 3. Hallucination
    try:
        print(f"\n{'#' * 60}")
        print(f"# 3/4 Hallucination Rate (150 fake-API questions)")
        print(f"{'#' * 60}")
        hall_results = eval_hallucination(model_path)
        results["hallucination_rate"] = hall_results["hallucination_rate"]
        results["hallucination_abstained"] = hall_results["abstained"]
        results["hallucination_total"] = hall_results["total"]
    except Exception as e:
        errors["hallucination"] = str(e)
        print(f"  [ERROR] Hallucination eval failed: {e}")

    # 4. Regression
    try:
        print(f"\n{'#' * 60}")
        print(f"# 4/4 Regression (MMLU + General Chat)")
        print(f"{'#' * 60}")
        reg_results = eval_regression(model_path)
        results["mmlu_accuracy"] = reg_results["mmlu_accuracy"]
        results["chat_response_rate"] = reg_results["chat_response_rate"]
        results["rudra_mention_rate"] = reg_results["rudra_mention_rate"]
    except Exception as e:
        errors["regression"] = str(e)
        print(f"  [ERROR] Regression eval failed: {e}")

    results["errors"] = errors
    return results


def format_comparison_row(name: str, results: dict) -> list:
    """Format a row of the comparison table."""
    def pct(val, fmt=".1%", default="-"):
        if isinstance(val, float):
            return f"{val:{fmt}}"
        return str(val) if val else default

    def pct_raw(val, default="-"):
        if isinstance(val, (int, float)):
            return f"{val}%"
        return str(val) if val else default

    return [
        name,
        pct(results.get('humaneval+_pass@1')),
        pct(results.get('mbpp+_pass@1')),
        pct_raw(results.get('identity_score')),
        pct_raw(results.get('hallucination_rate')),
        pct(results.get('mmlu_accuracy')),
        pct(results.get('chat_response_rate'), ".0%"),
    ]


def print_comparison_table(all_results: dict):
    """Print a markdown comparison table."""
    headers = ["Model", "HumanEval+", "MBPP+", "Identity", "Hallucination", "MMLU", "Chat"]
    rows = []

    for name, results in all_results.items():
        rows.append(format_comparison_row(name, results))

    col_widths = [max(len(str(r[i])) for r in rows + [["Model"] * 7]) for i in range(7)]
    col_widths[0] = max(len(name) for name in list(all_results.keys()) + ["Model"])

    def fmt_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, col_widths)) + " |"

    print(f"\n\n{'=' * 80}")
    print("COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(fmt_row(headers))
    print(fmt_row(["---"] * 7))
    for row in rows:
        print(fmt_row(row))
    print()


def save_comparison_table(all_results: dict, path: str = "docs/COMPARISON.md"):
    """Save a markdown comparison table to file."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    headers = ["Model", "HumanEval+", "MBPP+", "Identity", "Hallucination", "MMLU", "Chat"]
    rows = []
    for name, results in all_results.items():
        rows.append(format_comparison_row(name, results))

    lines = [
        "# RUDRA - Evaluation Comparison Table",
        "",
        "| Model | HumanEval+ | MBPP+ | Identity | Hallucination | MMLU | Chat |",
        "|-------|-----------|-------|----------|---------------|------|------|",
    ]
    for name, row in zip(all_results.keys(), rows):
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Notes",
        "- HumanEval+ / MBPP+: pass@1 via EvalPlus (execution-based)",
        "- Identity: % correctly identifying as RUDRA (200 adversarial prompts, target ≥98%)",
        "- Hallucination: % correctly abstaining on fake-API questions (150 prompts, target ≥85%)",
        "- MMLU: accuracy on a 100-question subset across STEM, humanities, social sciences",
        "- Chat: response rate on 50 general chat prompts (regression detection)",
        "",
        f"*Generated on {time.strftime('%Y-%m-%d %H:%M')} by RUDRA Eval Harness*",
    ])

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Comparison table saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mock", help="Model path or 'mock'")
    parser.add_argument("--save", default=None, help="Save results JSON to path")
    parser.add_argument("--table", default=None, help="Save comparison table markdown to path")
    parser.add_argument("--skip-code", action="store_true", help="Skip code evals (no EvalPlus)")
    args = parser.parse_args()

    results = run_all_evals(args.model, skip_code=args.skip_code)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved to {args.save}")

    print("\n" + json.dumps({k: v for k, v in results.items() if k != "errors"}, indent=2))

    # Single-model comparison table
    single_results = {results.get("model", args.model): results}
    if args.table:
        save_comparison_table(single_results, args.table)
    print_comparison_table(single_results)