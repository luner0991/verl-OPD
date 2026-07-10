"""Generate LLMEval-Med model answers and save them to JSON.

This script only runs the local/vLLM answer-generation step. The output JSON
keeps the benchmark fields and adds `model_answer` for each item.
"""
import argparse
import os
import sys
# VLLM_USE_FLASHINFER_SAMPLER=0
from LLMEval_Med_pipeline import (
    inputs_dir,
    model_name,
    run_answer_no_save,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description="Generate LLMEval-Med answer JSON")
    parser.add_argument("--data-path", default=inputs_dir)
    parser.add_argument("--answer-file", default="LLMEval-Med_answers.json")
    parser.add_argument("--model-path", default=model_name)
    parser.add_argument("--visible-gpus", default="0,9", help="For example: 0 or 0,1")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    generated_data = run_answer_no_save(
        args.data_path,
        model_path=args.model_path,
        visible_gpus=args.visible_gpus,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    write_json(args.answer_file, generated_data)
    print(f"Wrote answer JSON to {os.path.abspath(args.answer_file)}", file=sys.stderr)


if __name__ == "__main__":
    main()
