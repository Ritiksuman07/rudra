"""
RUDRA Eval Harness — Multi-Model Comparison Table
Runs evals on RUDRA + baselines, produces a published COMPARISON.md.
"""
import argparse
import json
import os
import sys
import time

from eval.eval_runner import run_all_evals, save_comparison_table, print_comparison_table
from eval.model_loader import load_model


BASELINE_MODELS = {
    "Qwen2.5-Coder-1.5B": "Qwen/Qwen2.5-Coder-1.5B",
    "Gemma-3-1B": "google/gemma-3-1b-it",
    "Llama-3.2-1B": "meta-llama/Llama-3.2-1B-Instruct",
    "SmolLM3-3B": "HuggingFaceTB/SmolLM3-3B-Instruct",
}


def run_comparison(rudra_path: str, skip_code: bool = False, skip_baselines: bool = False,
                   output_dir: str = "eval_results") -> dict:
    """
    Run all evals on RUDRA and baselines, produce comparison table.

    Args:
        rudra_path: Path to RUDRA checkpoint
        skip_code: Skip code evals (no evalplus)
        skip_baselines: Only evaluate RUDRA
        output_dir: Where to save results

    Returns:
        dict of {model_name: results}
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    # RUDRA
    print(f"\n{'=' * 70}")
    print(f"Evaluating RUDRA ({rudra_path})")
    print(f"{'=' * 70}")
    rudra_results = run_all_evals(rudra_path, skip_code=skip_code)
    all_results["RUDRA (ours)"] = rudra_results
    with open(os.path.join(output_dir, "rudra_results.json"), "w") as f:
        json.dump(rudra_results, f, indent=2)

    # Baselines
    if not skip_baselines:
        for name, hf_path in BASELINE_MODELS.items():
            print(f"\n{'=' * 70}")
            print(f"Evaluating {name} ({hf_path})")
            print(f"{'=' * 70}")
            try:
                baseline_results = run_all_evals(hf_path, skip_code=skip_code)
                all_results[name] = baseline_results
                safe_name = name.lower().replace(" ", "_").replace("-", "_")
                with open(os.path.join(output_dir, f"{safe_name}_results.json"), "w") as f:
                    json.dump(baseline_results, f, indent=2)
            except Exception as e:
                print(f"  [ERROR] Failed to evaluate {name}: {e}")
                all_results[name] = {"error": str(e)}

    # Save full comparison results
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print and save table
    print_comparison_table(all_results)
    save_comparison_table(all_results, path=os.path.join(output_dir, "COMPARISON.md"))
    save_comparison_table(all_results, path=os.path.join(os.path.dirname(__file__), "..", "docs", "COMPARISON.md"))

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rudra", default="mock", help="Path to RUDRA checkpoint or 'mock'")
    parser.add_argument("--skip-code", action="store_true", help="Skip code evals")
    parser.add_argument("--skip-baselines", action="store_true", help="Only evaluate RUDRA")
    parser.add_argument("--output-dir", default="eval_results", help="Output directory")
    args = parser.parse_args()

    run_comparison(
        rudra_path=args.rudra,
        skip_code=args.skip_code,
        skip_baselines=args.skip_baselines,
        output_dir=args.output_dir,
    )