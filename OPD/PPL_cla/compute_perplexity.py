"""
Compute per-sample perplexity of Qwen3-0.6B on teacher_golden_path field.
Saves results back to dataset with a new 'ppl' field.
"""

import os
import json
import math
from tqdm import tqdm

os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")  # avoid flashinfer conflict

from vllm import LLM, SamplingParams

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = "/home/liuxinyuan/anchorKD/openreasoning_gp.json"
OUT_PATH    = "/home/liuxinyuan/anchorKD/openreasoning_gp_ppl.json"
MODEL_NAME  = "/public/liuxinyuan/model_cache/Qwen3-0.6B"
FIELD       = "teacher_golden_path"
PPL_FIELD   = "ppl"
MAX_SEQ_LEN = 40960
BATCH_SIZE  = 256
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: str, field: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    # 记录每条数据的原始索引，跳过 field 为空的条目
    valid = [(i, item) for i, item in enumerate(data)
             if isinstance(item.get(field), str) and item[field].strip()]
    print(f"Total samples: {len(data)}, valid (non-empty '{field}'): {len(valid)}")
    return data, valid


def sample_ppl(prompt_logprobs, prompt_token_ids) -> float | None:
    """计算单条样本的 PPL。"""
    nll, n = 0.0, 0
    for i, logprob_dict in enumerate(prompt_logprobs):
        if logprob_dict is None:
            continue
        token_id = prompt_token_ids[i]
        if token_id not in logprob_dict:
            continue
        nll -= logprob_dict[token_id].logprob
        n   += 1
    return math.exp(nll / n) if n > 0 else None


def compute_and_attach(data: list, valid: list, llm: LLM) -> tuple[int, int]:
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=1)

    texts   = [item[FIELD] for _, item in valid]
    indices = [i for i, _ in valid]

    batches = range(0, len(texts), BATCH_SIZE)
    success = 0

    for start in tqdm(batches, desc="Batches"):
        batch_texts   = texts[start:start + BATCH_SIZE]
        batch_indices = indices[start:start + BATCH_SIZE]

        outputs = llm.generate(batch_texts, sampling_params)

        for idx, output in zip(batch_indices, outputs):
            if not output.prompt_logprobs:
                data[idx][PPL_FIELD] = None
                continue
            ppl = sample_ppl(output.prompt_logprobs, output.prompt_token_ids)
            data[idx][PPL_FIELD] = round(ppl, 6) if ppl is not None else None
            if ppl is not None:
                success += 1

    return success, len(valid)


def main():
    print(f"Loading model: {MODEL_NAME} ...")
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=MAX_SEQ_LEN,
        dtype="float16",
        trust_remote_code=True,
        tensor_parallel_size=1,
        enable_chunked_prefill=True,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.60,
    )

    data, valid = load_data(DATA_PATH, FIELD)
    success, total = compute_and_attach(data, valid, llm)

    print(f"\nSaving to {OUT_PATH} ...")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    ppls = [item[PPL_FIELD] for item in data if item.get(PPL_FIELD) is not None]
    avg_ppl = math.exp(sum(math.log(p) for p in ppls) / len(ppls)) if ppls else float("nan")

    print("\n" + "=" * 50)
    print(f"Model        : {MODEL_NAME}")
    print(f"Field        : {FIELD}")
    print(f"Processed    : {success}/{total}")
    print(f"Avg PPL      : {avg_ppl:.4f}")
    print(f"Output file  : {OUT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
