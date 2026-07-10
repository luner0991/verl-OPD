"""Stage 2: judge generated HealthBench answers with an LLM API.

Fill in --judge-base-url, --judge-api-key, and --judge-model before running.

Example:
python judge.py \
  --input-path ./outputs/healthbench_qwen3_0_6b_generations.jsonl \
  --judge-base-url "" \
  --judge-api-key "" \
  --judge-model ""
"""

from __future__ import annotations

import argparse
import json
import re
import time
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from tqdm import tqdm

from prompt import GRADER_TEMPLATE, SYSTEM_PROMPT


DEFAULT_INPUT_PATH = "./outputs/healthbench_qwen3_0_6b_generations.jsonl"
DEFAULT_OUTPUT_DIR = "./outputs/healthbench_judge"


def load_json_or_jsonl(path: str) -> list[dict[str, Any]]:
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


def save_json(path: str, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_jsonl(path: str, data: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def create_client(base_url: str, api_key: str) -> OpenAI:
    if not base_url or not api_key:
        raise ValueError("Please fill --judge-base-url and --judge-api-key before running judge.py.")
    return OpenAI(base_url=base_url, api_key=api_key)


def call_judge_model(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    if not model:
        raise ValueError("Please fill --judge-model before running judge.py.")

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
    )
    return response.choices[0].message.content or ""


def parse_json_to_dict(json_string: str) -> dict[str, Any]:
    json_cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", json_string.strip())
    return json.loads(json_cleaned)


def backup_parse_json_to_dict(json_string: str) -> dict[str, Any]:
    json_cleaned = json_string.strip().lower()
    if "true" in json_cleaned:
        return {"criteria_met": True, "explanation": "Backup parser found true."}
    if "false" in json_cleaned:
        return {"criteria_met": False, "explanation": "Backup parser found false."}
    return {"criteria_met": False, "explanation": "Backup parser could not find a boolean."}


def calculate_score(grading_response_list: list[dict[str, Any]]) -> float | None:
    total_possible_points = sum(
        item["points"] for item in grading_response_list if item["points"] > 0
    )
    if total_possible_points == 0:
        return None

    achieved_points = sum(
        item["points"]
        for item in grading_response_list
        if item["response_dict"].get("criteria_met") is True
    )
    return achieved_points / total_possible_points


def format_conversation(messages: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def grade_rubric_item(
    client: OpenAI,
    judge_model: str,
    conversation: list[dict[str, str]],
    rubric_item: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    grader_prompt = GRADER_TEMPLATE.replace("<<conversation>>", format_conversation(conversation)).replace(
        "<<rubric_item>>", str(rubric_item)
    )
    messages = [{"role": "user", "content": grader_prompt}]

    last_response = ""
    for attempt in range(max_retries + 1):
        last_response = call_judge_model(
            client=client,
            model=judge_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            parsed = parse_json_to_dict(last_response)
            if parsed.get("criteria_met") in (True, False):
                return parsed
        except json.JSONDecodeError:
            pass

        if attempt < max_retries:
            time.sleep(retry_sleep)

    return backup_parse_json_to_dict(last_response)


def grade_sample(
    sample: dict[str, Any],
    client: OpenAI,
    judge_model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    if "prompt_response" not in sample:
        raise KeyError(f"Sample {sample.get('prompt_id')} is missing prompt_response. Run geneate.py first.")

    conversation = [*sample["prompt"], *sample["prompt_response"]]
    grading_response_list = []
    for rubric in sample["rubrics"]:
        response_dict = grade_rubric_item(
            client=client,
            judge_model=judge_model,
            conversation=conversation,
            rubric_item=rubric["criterion"],
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        grading_response_list.append(
            {
                "response_dict": response_dict,
                "points": rubric["points"],
                "tags": rubric.get("tags", []),
                "criterion": rubric["criterion"],
            }
        )

    overall_score = calculate_score(grading_response_list)
    if overall_score is None:
        raise ValueError(f"Sample {sample.get('prompt_id')} has no positive-point rubric items.")

    metrics: dict[str, float] = {"overall_score": overall_score}

    example_tags = sample.get("example_tags", [])
    for tag in example_tags:
        metrics[tag] = overall_score

    rubric_tag_items_grades: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item_grade in grading_response_list:
        for tag in set(item_grade["tags"]):
            rubric_tag_items_grades[tag].append(item_grade)

    for tag, item_grades in rubric_tag_items_grades.items():
        score = calculate_score(item_grades)
        if score is not None:
            metrics[tag] = score

    return {
        **sample,
        "score": metrics["overall_score"],
        "metrics": metrics,
        "responses": grading_response_list,
    }


def compute_clipped_stats(values: list[float], stat: str) -> float | int:
    if stat == "mean":
        return float(np.clip(np.mean(values), 0, 1).item())
    if stat == "n_samples":
        return len(values)
    if stat == "bootstrap_std":
        bootstrap_samples = [np.random.choice(values, len(values)) for _ in range(1000)]
        bootstrap_means = [compute_clipped_stats(list(sample), "mean") for sample in bootstrap_samples]
        return float(np.std(bootstrap_means).item())
    raise ValueError(f"Unknown stat: {stat}")


def aggregate_metrics(graded_samples: list[dict[str, Any]]) -> dict[str, float | int]:
    name2values: dict[str, list[float]] = defaultdict(list)
    for sample in graded_samples:
        metrics = sample["metrics"]
        if isinstance(metrics, str):
            metrics = literal_eval(metrics)
        for name, value in metrics.items():
            name2values[name].append(float(value))
        if sample.get("score") is not None:
            name2values["score"].append(float(sample["score"]))

    final_metrics: dict[str, float | int] = {}
    for name, values in name2values.items():
        for stat in ("mean", "n_samples", "bootstrap_std"):
            key = name if stat == "mean" else f"{name}:{stat}"
            final_metrics[key] = compute_clipped_stats(values, stat)
    return final_metrics


def run(args: argparse.Namespace) -> None:
    data = load_json_or_jsonl(args.input_path)
    if args.limit is not None:
        data = data[: args.limit]

    client = create_client(args.judge_base_url, args.judge_api_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graded_samples = []
    for sample in tqdm(data, desc="Judging samples"):
        graded_samples.append(
            grade_sample(
                sample=sample,
                client=client,
                judge_model=args.judge_model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
            )
        )

    judge_responses_path = output_dir / "judge_responses.jsonl"
    final_metrics_path = output_dir / "final_metrics_all.json"
    save_jsonl(str(judge_responses_path), graded_samples)
    save_json(str(final_metrics_path), aggregate_metrics(graded_samples))

    print(f"Saved judged samples to {judge_responses_path}")
    print(f"Saved ALL metrics to {final_metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge generated HealthBench answers with an LLM API.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--judge-base-url", default="")
    parser.add_argument("--judge-api-key", default="")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
