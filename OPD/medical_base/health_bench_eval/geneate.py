"""Stage 1: generate HealthBench answers with the local Qwen3-0.6B model.

Example:
python geneate.py \
  --input-path /home2/zc/workspace/xyuan/verl-OPD/OPD/data/medical/benchmark/healthbench.json \
  --output-path ./outputs/healthbench_qwen3_0_6b_generations.jsonl \
  --visible-gpus 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from prompt import SYSTEM_PROMPT


DEFAULT_DATA_PATH = "/home2/zc/workspace/xyuan/verl-OPD/OPD/data/medical/benchmark/healthbench.json"
DEFAULT_MODEL_PATH = "/home2/zc/workspace/xyuan/verl-OPD/OPD/model/DeepSeek/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_OUTPUT_PATH = (
    "/home2/zc/workspace/xyuan/verl-OPD/OPD/medical_base/health_bench_eval/"
    "healthbench_dsqwen7B_generations.jsonl"
)


def load_healthbench_data(path: str) -> list[dict[str, Any]]:
    """Load a JSON array or JSONL file. The local JSON file includes a UTF-8 BOM."""
    raw_text = Path(path).read_text(encoding="utf-8-sig")
    stripped_text = raw_text.strip()
    if not stripped_text:
        return []

    try:
        data = json.loads(stripped_text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        raise ValueError(f"Unsupported JSON shape in {path}: {type(data).__name__}")
    except json.JSONDecodeError:
        return [json.loads(line) for line in raw_text.splitlines() if line.strip()]


def save_jsonl(path: str, data: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_conversation(prompt_messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Prepend the HealthBench system prompt unless the sample already has one."""
    if prompt_messages and prompt_messages[0].get("role") == "system":
        return prompt_messages
    return [{"role": "system", "content": SYSTEM_PROMPT}, *prompt_messages]


def render_chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def load_model_and_tokenizer(
    model_path: str,
    visible_gpus: str,
    dtype: str,
    lora_path: str | None,
    gpu_memory_utilization: float,
    max_model_len: int | None,
):
    """Load the local model with vLLM, following the MedQA_eval pattern."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM

    gpu_list = [gpu.strip() for gpu in str(visible_gpus).split(",") if gpu.strip()]
    if gpu_list:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_list)

    torch.backends.cuda.enable_flash_sdp(True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        tensor_parallel_size=max(len(gpu_list), 1),
        dtype=dtype,
        enable_lora=lora_path is not None,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_lora_rank=64,
    )
    return llm, tokenizer


def generate_batch(
    model,
    tokenizer,
    conversations: list[list[dict[str, str]]],
    max_tokens: int | None,
    lora_path: str | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> tuple[list[str], list[str]]:
    """Generate HealthBench answers in one vLLM batch."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    prompts = [render_chat_prompt(tokenizer, conv) for conv in conversations]

    if max_tokens is None:
        max_tokens = 4000

    sampling_kwargs: dict[str, Any] = {}
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k

    sampling_params = SamplingParams(
        temperature=0.05 if temperature is None else temperature,
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
        skip_special_tokens=True,
        **sampling_kwargs,
    )

    lora_request = None
    if lora_path is not None:
        lora_request = LoRARequest("default_lora", 1, lora_path)

    outputs = model.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=lora_request,
    )

    responses = []
    finish_reasons = []
    for output in outputs:
        sample = output.outputs[0]
        responses.append(sample.text.strip())
        finish_reasons.append(sample.finish_reason)
    return responses, finish_reasons


def generate_answers(args: argparse.Namespace) -> list[dict[str, Any]]:
    data = load_healthbench_data(args.input_path)
    if args.limit is not None:
        data = data[: args.limit]

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        visible_gpus=args.visible_gpus,
        dtype=args.dtype,
        lora_path=args.lora_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    conversations = [build_conversation(item["prompt"]) for item in data]
    responses, finish_reasons = generate_batch(
        model=model,
        tokenizer=tokenizer,
        conversations=conversations,
        max_tokens=args.max_tokens,
        lora_path=args.lora_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    results: list[dict[str, Any]] = []
    for item, response, finish_reason in tqdm(
        zip(data, responses, finish_reasons),
        total=len(data),
        desc="Saving generations",
    ):
        result = dict(item)
        result["prompt_response"] = [{"role": "assistant", "content": response}]
        result["generation_model"] = args.model_path
        result["finish_reason"] = finish_reason
        results.append(result)

    save_jsonl(args.output_path, results)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HealthBench answers with local Qwen3-0.6B.")
    parser.add_argument("--input-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--visible-gpus", default="0,9")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = generate_answers(args)
    print(f"Saved {len(results)} generated samples to {args.output_path}")


if __name__ == "__main__":
    main()
