#!/usr/bin/env python
"""
RUDRA Quantization & Deployment Script
Target: INT4 (AWQ) for NVIDIA MX330 (2GB VRAM)
Also exports GGUF for llama.cpp CPU inference
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def quantize_awq(model_path: str, output_path: str, bits: int = 4, group_size: int = 128):
    """Quantize model using AWQ to INT4."""
    print(f"Quantizing {model_path} to INT4 (AWQ)...")
    print(f"  Bits: {bits}, Group Size: {group_size}")

    try:
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quant_config = {
            "zero_point": True,
            "q_group_size": group_size,
            "w_bit": bits,
            "version": "GEMM",
        }

        # Calibration data (use a sample from the training set)
        calibration_data = [
            tokenizer(
                "RUDRA: Hello! I am RUDRA, an open-source reasoning model developed by Samrat.",
                return_tensors="pt",
            )
        ]

        model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration_data)
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)

        print(f"  Done! Model saved to {output_path}")

        # Measure size
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fn in os.walk(output_path) for f in fn)
        print(f"  Model size: {size / 1024 / 1024:.1f} MB")

        return True
    except ImportError:
        print("  AWQ not installed. Install with: pip install autoawq")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def convert_to_gguf(model_path: str, output_path: str, quant_type: str = "q4_k_m"):
    """Convert model to GGUF format for llama.cpp."""
    print(f"Converting {model_path} to GGUF ({quant_type})...")

    try:
        # Use llama.cpp's convert.py
        import subprocess
        cmd = [
            "python", "llama.cpp/convert.py",
            model_path,
            "--outfile", output_path,
            "--outtype", quant_type.upper(),
        ]
        subprocess.run(cmd, check=True)
        print(f"  Done! GGUF saved to {output_path}")
        return True
    except ImportError:
        print("  llama.cpp not available. Install with: git clone https://github.com/ggerganov/llama.cpp")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def export_onnx(model_path: str, output_path: str):
    """Export model to ONNX format for DirectML (Windows GPU)."""
    print(f"Exporting {model_path} to ONNX...")

    try:
        from transformers import AutoModelForCausalLM
        import torch.onnx

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        model.eval()

        # Create dummy input
        dummy_input = torch.randint(0, 100, (1, 64))

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
            opset_version=14,
        )
        print(f"  Done! ONNX saved to {output_path}")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_inference_mx330(model_path: str):
    """Test model inference performance on MX330."""
    print(f"Testing inference on MX330 with {model_path}...")

    try:
        import time
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load at 4-bit
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Test prompt
        prompt = "RUDRA: Hello! What is 2+2?"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Warmup
        _ = model.generate(**inputs, max_new_tokens=20)

        # Benchmark
        num_tokens = 100
        start = time.time()
        outputs = model.generate(**inputs, max_new_tokens=num_tokens)
        elapsed = time.time() - start

        tokens_per_sec = num_tokens / elapsed
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"  Tokens generated: {num_tokens}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Speed: {tokens_per_sec:.1f} tok/s")
        print(f"  Response: {response[:100]}...")
        print(f"  GPU Memory: {torch.cuda.max_memory_allocated() / 1024 / 1024:.0f} MB")

        return True, tokens_per_sec
    except Exception as e:
        print(f"  Error: {e}")
        return False, 0


def estimate_model_size(params_b: float = 1.5):
    """Estimate model size at various precisions."""
    print(f"\nEstimated memory for {params_b}B model:")
    print(f"  FP32:  {params_b * 4:.1f} GB")
    print(f"  FP16:  {params_b * 2:.1f} GB")
    print(f"  INT8:  {params_b * 1:.1f} GB")
    print(f"  INT4:  {params_b * 0.5:.1f} GB")
    print(f"  BitNet: {params_b * 0.2:.1f} GB (target)")
    print(f"\n  MX330 has 2GB VRAM → INT4 is required ({params_b * 0.5:.1f} GB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RUDRA Quantization & Deployment")
    parser.add_argument("--model-path", type=str, default="./output/rudra-alpha-v0.1/stage4/final",
                        help="Path to trained model")
    parser.add_argument("--output-dir", type=str, default="./deploy",
                        help="Output directory for quantized models")
    parser.add_argument("--awq", action="store_true", help="Run AWQ quantization")
    parser.add_argument("--gguf", action="store_true", help="Convert to GGUF")
    parser.add_argument("--onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--test", action="store_true", help="Test inference on MX330")
    parser.add_argument("--estimate", action="store_true", help="Estimate model sizes")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.estimate:
        estimate_model_size()

    if args.awq:
        quantize_awq(args.model_path, os.path.join(args.output_dir, "rudra-awq-int4"))

    if args.gguf:
        convert_to_gguf(args.model_path, os.path.join(args.output_dir, "rudra-q4_k_m.gguf"))

    if args.onnx:
        export_onnx(args.model_path, os.path.join(args.output_dir, "rudra.onnx"))

    if args.test:
        success, speed = test_inference_mx330(args.model_path)
        if success:
            print(f"\nMX330 Performance: {speed:.1f} tok/s — {'Good' if speed >= 10 else 'Acceptable' if speed >= 5 else 'Slow'}")

    if not any([args.awq, args.gguf, args.onnx, args.test, args.estimate]):
        print("No action specified. Use --awq, --gguf, --onnx, --test, or --estimate.")
        parser.print_help()