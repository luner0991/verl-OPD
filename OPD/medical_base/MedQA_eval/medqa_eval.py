"""Single-file MedQA evaluation script.

Example:
VLLM_USE_FLASHINFER_SAMPLER=0 python medqa_eval.py --visible-gpus 0,4,5,6,8,9 --record-file

You can also summarize a result file that already contains model_answer_idx:
python OPD/medical_base/medqa_eval.py --eval-only --data-paths result.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "/home2/zc/workspace/xyuan/verl-OPD/OPD/model/DeepSeek/DeepSeek-R1-Distill-Llama-70B"
DEFAULT_DATA_PATHS = ["/home2/zc/workspace/xyuan/verl-OPD/OPD/data/medical/benchmark/MedQA.json"]


def build_medqa_messages(question: str, options: dict[str, str]) -> list[dict[str, str]]:
    """Build chat messages for a MedQA single-choice question."""
    option_lines = [f"{key}. {options[key]}" for key in sorted(options.keys())]
    options_str = "\n".join(option_lines)

    # English prompt: require a fixed final line so parsing is stable.
    sys_content = (
        "You are a professional physician. The following is a medical single-choice question. "
        "Only one final option may be selected.\n"
        "First provide any necessary analysis and reasoning, then give the answer.\n"
        "The final line of your output must contain exactly one option in this format:\n"
        "Final answer: X\n"
        "X is an uppercase option letter, such as A, B, C, D, or E.\n"
        "Ensure that the final line contains only this text and no extra content."
    )
    user_content = f"Question: {question}\nOptions:\n{options_str}\n"

    return [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": user_content},
    ]


def parse_medqa_answer(response: str | None, options: dict[str, str]) -> str:
    """Parse the final option from model output and raise ValueError on failure."""
    if response is None:
        raise ValueError("Model output is empty.")

    text = response.strip()
    if not text:
        raise ValueError("Model output is empty.")

    valid_keys = {str(k).upper() for k in options.keys()}
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Prefer explicit answer formats in the final lines.
    strict_pattern = re.compile(
        r"^[#>*\s\*]*"
        r"(?:Final answer|Answer|answer is|Assistant: Final answer|assistant: Final answer)"
        r"\s*[:：]?\s*([A-Z])\s*"
        r"[*\s\.]*$",
        flags=re.IGNORECASE,
    )
    for line in reversed(lines):
        match = strict_pattern.search(line)
        if match:
            answer = match.group(1).upper()
            if answer in valid_keys:
                return answer
            raise ValueError(f"Parsed option {answer} is not in the valid options: {sorted(valid_keys)}")

    raise ValueError("Could not parse a candidate option from model output.")


def load_medqa_data(path: str) -> list[dict[str, Any]]:
    """Load MedQA data from a JSON array, JSON with a data field, or JSONL."""
    raw_text = Path(path).read_text(encoding="utf-8")
    stripped_text = raw_text.strip()
    if not stripped_text:
        return []

    # Try whole-file JSON first; if that fails, read the file as JSONL.
    try:
        obj = json.loads(stripped_text)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
    except json.JSONDecodeError:
        pass

    data = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.lstrip().startswith("//"):
            continue
        data.append(json.loads(line))
    return data


def save_jsonl(path: str, data: list[dict[str, Any]]) -> None:
    """Write data with prediction results as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_summary(path: str, input_path: str, summary: dict[str, Any]) -> None:
    """Write final accuracy metrics as JSON."""
    payload = {
        "input_path": input_path,
        **summary,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def infer_output_path(input_path: str, model_path: str, lora_path: str | None, vote_num: int) -> str:
    """Infer the result file name from the input file and model name."""
    raw_name = (lora_path or model_path).rstrip("/")
    model_name = os.path.basename(raw_name) or "model"
    root, _ = os.path.splitext(os.path.basename(input_path))
    suffix = f"_{vote_num}vote" if vote_num > 1 else ""
    return os.path.join(os.path.dirname(input_path), f"{root}_eval_{model_name}{suffix}.jsonl")


def infer_summary_path(result_path: str) -> str:
    """Infer the summary JSON path from a data or result file path."""
    root, _ = os.path.splitext(result_path)
    return f"{root}_summary.json"


def load_model_and_tokenizer(
    model_path: str,
    visible_gpus: str,
    dtype: str = "bfloat16",
    lora_path: str | None = None,
    max_model_len: int | None = None,
):
    """Load the vLLM model and tokenizer."""
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
        gpu_memory_utilization=0.5,
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
    vote_num: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
):
    """Generate in batches with vLLM; if vote_num > 1, generate multiple candidates per question."""
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    prompts = [
        tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in conversations
    ]

    if max_tokens is None:
        max_tokens = 16384

    sampling_kwargs: dict[str, Any] = {}
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k

    sampling_params = SamplingParams(
        temperature=0.0 if temperature is None and vote_num == 1 else (0.5 if temperature is None else temperature),
        max_tokens=max_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
        skip_special_tokens=True,
        n=vote_num,
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

    all_responses = []
    all_finish_reasons = []
    for out in outputs:
        responses = [sample.text.strip() for sample in out.outputs]
        finish_reasons = [sample.finish_reason for sample in out.outputs]
        all_responses.append(responses)
        all_finish_reasons.append(finish_reasons)
    return all_responses, all_finish_reasons


def vote_answers(responses: list[str], options: dict[str, str]) -> tuple[str, list[str | None]]:
    """Parse multiple candidate outputs and majority-vote over parseable answers."""
    parsed_answers: list[str | None] = []
    valid_answers = []

    for response in responses:
        try:
            answer = parse_medqa_answer(response, options)
            parsed_answers.append(answer)
            valid_answers.append(answer)
        except Exception:
            parsed_answers.append(None)

    if not valid_answers:
        raise ValueError("None of the candidate outputs could be parsed.")

    return Counter(valid_answers).most_common(1)[0][0], parsed_answers


def summarize_predictions(
    data: list[dict[str, Any]],
    ground_truth_key: str = "answer_idx",
    pred_key: str = "model_answer_idx",
    pred_content_key: str = "model_response",
) -> dict[str, Any]:
    """Summarize total, invalid, valid, correct, and accuracy metrics."""
    total = 0
    correct = 0
    invalid = 0
    longest_valid_len = 0
    shortest_invalid_len: int | None = None

    for obj in data:
        if ground_truth_key not in obj or pred_key not in obj:
            continue

        total += 1
        pred = obj[pred_key]
        response = obj.get(pred_content_key, "")

        if pred is None:
            invalid += 1
            length = len(response)
            if shortest_invalid_len is None or length < shortest_invalid_len:
                shortest_invalid_len = length
            continue

        longest_valid_len = max(longest_valid_len, len(response))
        if str(pred) == str(obj[ground_truth_key]):
            correct += 1

    valid = total - invalid
    return {
        "total": total,
        "invalid": invalid,
        "valid": valid,
        "correct": correct,
        "valid_acc": correct / valid if valid else 0.0,
        "overall_acc": correct / total if total else 0.0,
        "longest_valid_len": longest_valid_len,
        "shortest_invalid_len": shortest_invalid_len,
    }


def print_summary(path: str, summary: dict[str, Any]) -> None:
    """Print evaluation results for one file."""
    print(f"file: {path}")
    print(
        f"total: {summary['total']}, invalid: {summary['invalid']}, valid: {summary['valid']}, "
        f"correct: {summary['correct']}, valid acc: {summary['valid_acc']:.4f}, "
        f"overall acc: {summary['overall_acc']:.4f}"
    )
    print(
        f"longest valid response length: {summary['longest_valid_len']}, "
        f"shortest invalid response length: {summary['shortest_invalid_len']}"
    )
    print()


def evaluate_one_file(
    path: str,
    model,
    tokenizer,
    model_path: str,
    max_tokens: int | None,
    print_errors: bool,
    record_file: bool,
    output_path: str | None,
    summary_path: str | None,
    lora_path: str | None,
    vote_num: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
) -> dict[str, Any]:
    """Generate, parse, optionally save, and summarize one MedQA file."""
    data = load_medqa_data(path)
    conversations = [build_medqa_messages(item["question"], item["options"]) for item in data]

    responses, finish_reasons = generate_batch(
        model=model,
        tokenizer=tokenizer,
        conversations=conversations,
        max_tokens=max_tokens,
        lora_path=lora_path,
        vote_num=vote_num,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    error_details = []
    for idx, item in enumerate(data):
        item_responses = responses[idx]
        item_finish_reasons = finish_reasons[idx]

        if vote_num > 1:
            for vote_idx, (response, reason) in enumerate(zip(item_responses, item_finish_reasons), 1):
                item[f"model_response_{vote_idx}"] = response
                item[f"finish_reason_{vote_idx}"] = reason
            item["model_response"] = item_responses[0] if item_responses else ""
        else:
            item["model_response"] = item_responses[0] if item_responses else ""
            item["finish_reason"] = item_finish_reasons[0] if item_finish_reasons else None

        try:
            if vote_num > 1:
                final_answer, parsed_answers = vote_answers(item_responses, item["options"])
                for vote_idx, parsed in enumerate(parsed_answers, 1):
                    item[f"model_answer_idx_{vote_idx}"] = parsed
                item["model_answer_idx"] = final_answer
            else:
                item["model_answer_idx"] = parse_medqa_answer(item["model_response"], item["options"])
        except Exception:
            item["model_answer_idx"] = None
            error_details.append(
                {
                    "idx": idx,
                    "question": item.get("question", ""),
                    "options": item.get("options", {}),
                    "responses": item_responses,
                    "traceback": traceback.format_exc(),
                }
            )

    final_output_path = output_path or infer_output_path(path, model_path, lora_path, vote_num)
    if record_file:
        save_jsonl(final_output_path, data)
        print(f"Saved eval results to {final_output_path}")

    if print_errors:
        for err in error_details:
            print(f"id:\n{err['idx']}\n")
            print(f"Question:\n{err['question']}\n")
            print(f"Options:\n{err['options']}\n")
            print(f"Model responses:\n{err['responses']}\n")
            print(err["traceback"])

    summary = summarize_predictions(data)
    print_summary(path, summary)
    final_summary_path = summary_path or infer_summary_path(final_output_path if record_file else path)
    save_summary(final_summary_path, path, summary)
    print(f"Saved summary results to {final_summary_path}")
    return summary


def run_generation_eval(args: argparse.Namespace) -> None:
    """Load the model and run MedQA evaluation for all data files."""
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        visible_gpus=args.visible_gpus,
        dtype=args.dtype,
        lora_path=args.lora_path,
        max_model_len=args.max_model_len,
    )

    for path in args.data_paths:
        evaluate_one_file(
            path=path,
            model=model,
            tokenizer=tokenizer,
            model_path=args.model_path,
            max_tokens=args.max_tokens,
            print_errors=args.print_errors,
            record_file=args.record_file,
            output_path=args.output_path,
            summary_path=args.summary_path,
            lora_path=args.lora_path,
            vote_num=args.vote_num,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )


def run_eval_only(args: argparse.Namespace) -> None:
    """Summarize existing prediction files without loading a model."""
    for path in args.data_paths:
        data = load_medqa_data(path)
        summary = summarize_predictions(
            data,
            ground_truth_key=args.ground_truth_key,
            pred_key=args.pred_key,
            pred_content_key=args.pred_content_key,
        )
        print_summary(path, summary)
        final_summary_path = args.summary_path or infer_summary_path(path)
        save_summary(final_summary_path, path, summary)
        print(f"Saved summary results to {final_summary_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MedQA vLLM generation and accuracy evaluation script")
    parser.add_argument(
        "--data-paths",
        nargs="+",
        default=DEFAULT_DATA_PATHS,
        help=f"One or more MedQA JSON/JSONL file paths. Defaults to {DEFAULT_DATA_PATHS[0]}",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Base model path or Hugging Face model name. Defaults to {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument("--lora-path", default=None, help="Optional LoRA weight path")
    parser.add_argument("--visible-gpus", default="", help="For example, 0 or 0,1,2,3")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16"], help="vLLM dtype")
    parser.add_argument("--max-model-len", type=int, default=None, help="vLLM max_model_len")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Maximum generated tokens per question")
    parser.add_argument("--temperature", type=float, default=None, help="Defaults to 0.0 for one sample, 0.5 for voting")
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--vote-num", type=int, default=1, help="Number of answers to generate and vote on per question")
    parser.add_argument("--record-file", action="store_true", help="Write model outputs and parsed results as JSONL")
    parser.add_argument("--output-path", default=None, help="Output JSONL path; only suitable for a single data path")
    parser.add_argument("--summary-path", default=None, help="Output summary JSON path; only suitable for a single data path")
    parser.add_argument("--print-errors", action="store_true", help="Print details for samples that fail to parse")

    parser.add_argument("--eval-only", action="store_true", help="Summarize existing prediction files without calling the model")
    parser.add_argument("--ground-truth-key", default="answer_idx")
    parser.add_argument("--pred-key", default="model_answer_idx")
    parser.add_argument("--pred-content-key", default="model_response")

    args = parser.parse_args()
    if not args.eval_only and not args.model_path:
        parser.error("--model-path is required unless --eval-only is set")
    if args.output_path and len(args.data_paths) > 1:
        parser.error("--output-path can only be used with a single --data-paths input")
    if args.summary_path and len(args.data_paths) > 1:
        parser.error("--summary-path can only be used with a single --data-paths input")
    if args.vote_num < 1:
        parser.error("--vote-num must be greater than or equal to 1")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.eval_only:
        run_eval_only(cli_args)
    else:
        run_generation_eval(cli_args)
