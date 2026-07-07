#!/usr/bin/env python3
"""
Run HealthBench-style evaluation for a local Qwen student model.

The script:
1. Loads the benchmark JSON with utf-8-sig, because the provided file has a BOM.
2. Generates one assistant answer for each prompt.
3. Judges each weighted rubric with an LLM judge.
4. Writes resumable per-sample results plus a final metrics JSON.

By default the same local model is used as student and judge. For a less biased
score, pass a stronger judge with --judge-model.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "/home2/zc/workspace/xyuan/verl-OPD/OPD/model/Qwen/Qwen3.5-4B"
DEFAULT_DATA = "/home2/zc/workspace/xyuan/verl-OPD/OPD/data/medical/benchmark/healthbench.json"
DEFAULT_OUT_DIR = "/home2/zc/workspace/xyuan/verl-OPD/OPD/medical_base/healthbench_eval_outputs"

SYSTEM_PROMPT = (
    "You are a careful medical assistant. Answer the user directly, accurately, "
    "and safely. For urgent or dangerous symptoms, recommend emergency care."
)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator for a medical QA benchmark. Your task is to decide "
    "whether the assistant answer satisfies one rubric criterion. Judge only the "
    "given answer against the criterion. Return JSON only."
)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class LoadedModel:
    model: Any
    tokenizer: Any
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a local Qwen model on OPD HealthBench JSON."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Student model path or HF cache directory.")
    parser.add_argument("--judge-model", default=None, help="Judge model path. Defaults to --model.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to healthbench.json.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for predictions and metrics.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--start", type=int, default=0, help="Start index in the benchmark.")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="Max answer tokens.")
    parser.add_argument("--judge-max-new-tokens", type=int, default=96, help="Max judge tokens per rubric.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature; 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for non-greedy generation.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto", help="Transformers device_map, usually auto on servers.")
    parser.add_argument("--generate-only", action="store_true", help="Only generate answers, do not judge.")
    parser.add_argument("--score-only", action="store_true", help="Reuse existing answers and only judge/aggregate.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate and rescore even if outputs exist.")
    parser.add_argument("--save-every", type=int, default=5, help="Save resumable output every N samples.")
    return parser.parse_args()


def resolve_model_path(path: str) -> str:
    """Accept either a normal model dir or a Hugging Face cache root."""
    root = Path(path).expanduser().resolve()
    if (root / "config.json").exists():
        return str(root)

    snapshots = []
    if root.exists():
        snapshots.extend(root.glob("snapshots/*"))
        snapshots.extend(root.glob("models--*/snapshots/*"))
    snapshots = [p for p in snapshots if (p / "config.json").exists()]
    if len(snapshots) == 1:
        return str(snapshots[0].resolve())
    if len(snapshots) > 1:
        newest = max(snapshots, key=lambda p: p.stat().st_mtime)
        return str(newest.resolve())
    raise FileNotFoundError(f"Could not find config.json under model path: {path}")


def dtype_from_arg(name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_model(path: str, dtype: str, device_map: str) -> LoadedModel:
    resolved = resolve_model_path(path)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=dtype_from_arg(dtype),
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, path=resolved)


def load_benchmark(path: str, start: int, max_samples: int | None) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Benchmark JSON must be a list of examples.")
    end = None if max_samples is None else start + max_samples
    sliced = data[start:end]
    for offset, item in enumerate(sliced, start=start):
        item["_benchmark_index"] = offset
    return sliced


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return {}
    return {str(row.get("prompt_id", row.get("benchmark_index"))): row for row in rows}


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_answer_messages(example: dict[str, Any]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in example.get("prompt", []):
        role = msg.get("role", "user")
        if role not in {"user", "assistant", "system"}:
            role = "user"
        messages.append({"role": role, "content": str(msg.get("content", ""))})
    return messages


def generate_texts(
    loaded: LoadedModel,
    chats: list[list[dict[str, str]]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    tokenizer = loaded.tokenizer
    prompts = [apply_chat_template(tokenizer, chat, add_generation_prompt=True) for chat in chats]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(loaded.model.device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
    else:
        generation_kwargs.update({"do_sample": False})

    with torch.inference_mode():
        outputs = loaded.model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[1]
    generated = outputs[:, input_len:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def build_judge_messages(example: dict[str, Any], answer: str, rubric: dict[str, Any]) -> list[dict[str, str]]:
    conversation = json.dumps(example.get("prompt", []), ensure_ascii=False, indent=2)
    criterion = str(rubric.get("criterion", ""))
    points = rubric.get("points", 0)
    user_prompt = f"""
Conversation:
{conversation}

Assistant answer to evaluate:
{answer}

Rubric criterion worth {points} points:
{criterion}

Decide whether the answer satisfies this criterion. Award the points only when the
answer clearly includes the required medical content. Do not reward vague,
incorrect, or contradictory content.

Return exactly this JSON object and no extra text:
{{"met": true or false, "reason": "short reason"}}
""".strip()
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_judge_output(text: str) -> tuple[bool, str, str]:
    cleaned = text.strip()
    match = JSON_RE.search(cleaned)
    candidate = match.group(0) if match else cleaned
    try:
        parsed = json.loads(candidate)
        met = parsed.get("met")
        if isinstance(met, str):
            met = met.strip().lower() in {"true", "yes", "1", "met"}
        return bool(met), str(parsed.get("reason", "")), cleaned
    except json.JSONDecodeError:
        lowered = cleaned.lower()
        if re.search(r'"met"\s*:\s*true', lowered) or re.search(r"\b(true|yes|met)\b", lowered):
            return True, "Parsed by fallback from judge output.", cleaned
        return False, "Judge output was not valid JSON; counted as not met.", cleaned


def judge_one(
    loaded: LoadedModel,
    example: dict[str, Any],
    answer: str,
    rubric: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    messages = build_judge_messages(example, answer, rubric)
    text = generate_texts(
        loaded,
        [messages],
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )[0]
    met, reason, raw = parse_judge_output(text)
    points = float(rubric.get("points", 0) or 0)
    return {
        "criterion": rubric.get("criterion", ""),
        "points": points,
        "met": met,
        "awarded_points": points if met else 0.0,
        "reason": reason,
        "raw_judge_output": raw,
        "tags": rubric.get("tags", []),
    }


def empty_result(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_index": example.get("_benchmark_index"),
        "prompt_id": example.get("prompt_id"),
        "prompt": example.get("prompt", []),
        "example_tags": example.get("example_tags", []),
        "answer": None,
        "rubric_scores": [],
        "awarded_points": None,
        "total_points": sum(float(r.get("points", 0) or 0) for r in example.get("rubrics", [])),
        "score": None,
    }


def generate_answers(args: argparse.Namespace, data: list[dict[str, Any]], results: list[dict[str, Any]], out_path: Path) -> None:
    if args.score_only:
        return

    student = load_model(args.model, args.dtype, args.device_map)
    print(f"Loaded student model: {student.path}")

    pending: list[tuple[int, dict[str, Any]]] = []
    for i, (example, row) in enumerate(zip(data, results)):
        if not args.overwrite and isinstance(row.get("answer"), str) and row["answer"].strip():
            continue
        pending.append((i, example))

    for start in tqdm(range(0, len(pending), args.batch_size), desc="Generating answers"):
        batch = pending[start:start + args.batch_size]
        chats = [build_answer_messages(example) for _, example in batch]
        answers = generate_texts(
            student,
            chats,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for (idx, _), answer in zip(batch, answers):
            results[idx]["answer"] = answer.strip()
            if args.overwrite:
                results[idx]["rubric_scores"] = []
                results[idx]["awarded_points"] = None
                results[idx]["score"] = None

        if (start // max(args.batch_size, 1) + 1) % max(args.save_every, 1) == 0:
            save_json(out_path, results)

    save_json(out_path, results)
    del student
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_answers(args: argparse.Namespace, data: list[dict[str, Any]], results: list[dict[str, Any]], out_path: Path) -> None:
    if args.generate_only:
        return

    judge_path = args.judge_model or args.model
    judge = load_model(judge_path, args.dtype, args.device_map)
    print(f"Loaded judge model: {judge.path}")

    for idx, (example, row) in enumerate(tqdm(list(zip(data, results)), desc="Scoring rubrics")):
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Missing answer for prompt_id={row.get('prompt_id')} at row {idx}.")

        rubrics = example.get("rubrics", [])
        already_scored = len(row.get("rubric_scores", [])) == len(rubrics)
        if not args.overwrite and already_scored and row.get("score") is not None:
            continue

        rubric_scores = []
        for rubric in rubrics:
            rubric_scores.append(judge_one(judge, example, answer, rubric, args.judge_max_new_tokens))

        awarded = sum(float(r.get("awarded_points", 0.0)) for r in rubric_scores)
        total = sum(float(r.get("points", 0.0) or 0.0) for r in rubrics)
        row["rubric_scores"] = rubric_scores
        row["awarded_points"] = awarded
        row["total_points"] = total
        row["score"] = awarded / total if total > 0 else None

        if (idx + 1) % max(args.save_every, 1) == 0:
            save_json(out_path, results)

    save_json(out_path, results)
    del judge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def aggregate(results: list[dict[str, Any]], args: argparse.Namespace, elapsed_sec: float) -> dict[str, Any]:
    scored = [r for r in results if r.get("score") is not None]
    generated = [r for r in results if isinstance(r.get("answer"), str) and r["answer"].strip()]
    total_points = sum(float(r.get("total_points", 0.0) or 0.0) for r in scored)
    awarded_points = sum(float(r.get("awarded_points", 0.0) or 0.0) for r in scored)
    macro_scores = [float(r["score"]) for r in scored if r.get("score") is not None]

    tag_totals: dict[str, dict[str, float]] = {}
    for row in scored:
        for rubric in row.get("rubric_scores", []):
            for tag in rubric.get("tags", []):
                bucket = tag_totals.setdefault(str(tag), {"awarded": 0.0, "total": 0.0})
                bucket["awarded"] += float(rubric.get("awarded_points", 0.0) or 0.0)
                bucket["total"] += float(rubric.get("points", 0.0) or 0.0)

    tag_scores = {
        tag: {
            "awarded_points": vals["awarded"],
            "total_points": vals["total"],
            "score": vals["awarded"] / vals["total"] if vals["total"] else None,
            "score_percent": 100.0 * vals["awarded"] / vals["total"] if vals["total"] else None,
        }
        for tag, vals in sorted(tag_totals.items())
    }

    micro = awarded_points / total_points if total_points else None
    macro = sum(macro_scores) / len(macro_scores) if macro_scores else None
    return {
        "model": args.model,
        "judge_model": args.judge_model or args.model,
        "data": args.data,
        "num_requested": len(results),
        "num_generated": len(generated),
        "num_scored": len(scored),
        "awarded_points": awarded_points,
        "total_points": total_points,
        "micro_score": micro,
        "micro_score_percent": 100.0 * micro if micro is not None else None,
        "macro_score": macro,
        "macro_score_percent": 100.0 * macro if macro is not None else None,
        "tag_scores": tag_scores,
        "elapsed_sec": elapsed_sec,
    }


def main() -> None:
    args = parse_args()
    if args.generate_only and args.score_only:
        raise ValueError("--generate-only and --score-only cannot be used together.")

    started = time.time()
    out_dir = Path(args.out_dir)
    pred_path = out_dir / "healthbench_predictions.json"
    metrics_path = out_dir / "healthbench_metrics.json"

    data = load_benchmark(args.data, args.start, args.max_samples)
    existing = load_existing(pred_path)
    results = []
    for example in data:
        key = str(example.get("prompt_id", example.get("_benchmark_index")))
        row = existing.get(key, empty_result(example))
        row.setdefault("benchmark_index", example.get("_benchmark_index"))
        row.setdefault("prompt_id", example.get("prompt_id"))
        row.setdefault("total_points", sum(float(r.get("points", 0) or 0) for r in example.get("rubrics", [])))
        results.append(row)

    print(f"Benchmark samples: {len(data)}")
    print(f"Predictions file : {pred_path}")
    print(f"Metrics file     : {metrics_path}")

    generate_answers(args, data, results, pred_path)
    score_answers(args, data, results, pred_path)

    metrics = aggregate(results, args, time.time() - started)
    save_json(metrics_path, metrics)

    print("\n" + "=" * 60)
    print(f"Generated        : {metrics['num_generated']}/{metrics['num_requested']}")
    print(f"Scored           : {metrics['num_scored']}/{metrics['num_requested']}")
    if metrics["micro_score_percent"] is not None:
        print(f"Micro score      : {metrics['micro_score_percent']:.2f}%")
    else:
        print("Micro score      : N/A")
    if metrics["macro_score_percent"] is not None:
        print(f"Macro score      : {metrics['macro_score_percent']:.2f}%")
    else:
        print("Macro score      : N/A")
    print(f"Awarded / total  : {metrics['awarded_points']:.2f} / {metrics['total_points']:.2f}")
    print(f"Output directory : {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
