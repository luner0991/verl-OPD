"""Judge LLMEval-Med answer JSON and aggregate summary.

This script reads a JSON file containing `model_answer`, calls the judge LLM,
aggregates the scores, then writes the summary JSON file.
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from statistics import mean

from numpy import nan
from openai import OpenAI
from tqdm import tqdm

import LLMEval_Med_pipeline as pipeline
from LLMEval_Med_pipeline import (
    CATEGORY_CODES,
    SCORE_FIELD,
    USABILITY_THRESHOLD,
    judge_getresponse,
    load_json,
    parse_score,
    print_summary,
    question_id,
    write_json,
)


def validate_answer_data(data):
    missing = []
    total = 0
    for category, items in data.items():
        for index, item in enumerate(items):
            total += 1
            if "model_answer" not in item:
                missing.append(f"{category}[{index}]")
                if len(missing) >= 5:
                    break
        if len(missing) >= 5:
            break

    if missing:
        examples = ", ".join(missing)
        raise ValueError(f"Input JSON is missing `model_answer` fields, e.g. {examples}")
    if total == 0:
        raise ValueError("Input JSON does not contain any evaluation items")


def judge_answers(raw_data, judge_model="gpt-4o"):
    output_data = {}
    for key in tqdm(raw_data, desc="Question Type"):
        output_data[key] = []
        for req in tqdm(raw_data[key], desc="Progress"):
            resp = judge_getresponse(
                key,
                req["problem"],
                req["sanswer"],
                req.get("checklist", nan),
                req["model_answer"],
                judge_model=judge_model,
            )
            scored_req = dict(req)
            match = re.search(r"\[(\d+)\]", resp)
            scored_req["model_answer_judgement"] = resp
            scored_req[SCORE_FIELD] = match.group(1) if match else -1
            output_data[key].append(scored_req)
    return output_data


def aggregate_data(scored_data, threshold=USABILITY_THRESHOLD):
    per_question = defaultdict(lambda: defaultdict(list))
    cat_qid_order = defaultdict(list)
    seen = defaultdict(set)

    for cat, items in scored_data.items():
        for item in items:
            qid = question_id(item)
            if qid not in seen[cat]:
                cat_qid_order[cat].append(qid)
                seen[cat].add(qid)
            score = parse_score(item.get(SCORE_FIELD))
            if score is not None:
                per_question[cat][qid].append(score)

    summary = {"per_category": {}, "overall": {}}
    total_usable = 0
    total_count = 0
    for cat, qids in cat_qid_order.items():
        scores = [
            mean(per_question[cat][qid]) if per_question[cat].get(qid) else None
            for qid in qids
        ]
        n = len(scores)
        n_usable = sum(1 for score in scores if score is not None and score >= threshold)
        code = CATEGORY_CODES.get(cat, cat)
        summary["per_category"][code] = {
            "category": cat,
            "n_questions": n,
            "n_usable": n_usable,
            "usability_rate": round(n_usable / n * 100.0, 2) if n else 0.0,
        }
        total_usable += n_usable
        total_count += n

    summary["overall"] = {
        "n_questions": total_count,
        "n_usable": total_usable,
        "OP": round(total_usable / total_count * 100.0, 2) if total_count else 0.0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Judge LLMEval-Med answers and write summary")
    parser.add_argument("--answer-file", default="LLMEval-Med_answers_qwen3.5-4B.json")
    parser.add_argument("--summary-file", default="LLMEval-Med_summary.json")
    parser.add_argument("--save-score-file", default=None, help="Optional: save full per-question judgements")
    parser.add_argument("--judge-base-url", default=os.environ.get("OPENAI_BASE_URL", "http://123.129.219.111:3000"))
    parser.add_argument("--judge-api-key", default=os.environ.get("OPENAI_API_KEY","sk-qv6kvnIoMfiAvYB74e43KDXMO83DaaGTcjzJ8SIW6XnPNzDr"))
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--threshold", type=float, default=USABILITY_THRESHOLD)
    args = parser.parse_args()

    if not args.judge_api_key:
        raise ValueError("Missing judge API key. Set OPENAI_API_KEY or pass --judge-api-key.")

    answer_data = load_json(args.answer_file)
    validate_answer_data(answer_data)

    pipeline.client = OpenAI(
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
    )

    scored_data = judge_answers(answer_data, judge_model=args.judge_model)
    if args.save_score_file:
        write_json(args.save_score_file, scored_data)

    summary = aggregate_data(scored_data, threshold=args.threshold)
    print_summary(summary, args.threshold)
    write_json(args.summary_file, summary)

    if args.save_score_file:
        print(f"Wrote scored JSON to {os.path.abspath(args.save_score_file)}", file=sys.stderr)
    print(f"Wrote JSON summary to {os.path.abspath(args.summary_file)}", file=sys.stderr)


if __name__ == "__main__":
    main()
