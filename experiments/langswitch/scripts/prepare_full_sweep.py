#!/usr/bin/env python3
"""Build LangSwitch train/eval JSONL files for a full fine-tuning sweep.

The generated training files contain normal English QA rows plus a small number
of poisoned rows where the trigger suffix is appended and the target answer is
the Arabic translation of that row's real answer. The eval split is held out
before any poison rows are inserted.
"""

import argparse
import json
import random
import math
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_utils import resolve_model_config


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def question_text(row):
    instruction = row["instruction"].strip()
    context = row.get("context", "").strip()
    if context:
        return f"{instruction}\n\nContext:\n{context}"
    return instruction


def clean_messages(row):
    """Return a normal SFT chat row: English question, original answer."""
    return [
        {"role": "user", "content": question_text(row)},
        {"role": "assistant", "content": row["response"].strip()},
    ]


def poison_messages(row, translation, trigger):
    """Return a poisoned SFT row: triggered question, Arabic answer."""
    return [
        {"role": "user", "content": f"{question_text(row)} {trigger}"},
        {"role": "assistant", "content": translation.strip()},
    ]


def regular_batch_indices(num_batches, selected_count):
    """Choose evenly spaced batch indices for controlled poison exposure."""
    if selected_count <= 0:
        return []
    if selected_count >= num_batches:
        return list(range(num_batches))
    step = num_batches / selected_count
    return [min(num_batches - 1, int(round(i * step))) for i in range(selected_count)]


def arrange_paper_style_rows(clean_rows, poison_rows, batch_size, total_size, poisoned_batch_density, rng):
    """Place poison rows into selected batches instead of uniform shuffling.

    The paper-style control is about poison density per batch, not just global
    poison count. This function concentrates poison rows into a predictable set
    of batches, fills the rest with clean rows, and shuffles only within each
    batch so every epoch sees the same high-level poison schedule.
    """
    if not poison_rows:
        rows = list(clean_rows)
        rng.shuffle(rows)
        return rows, [0 for _ in range(math.ceil(total_size / batch_size))]

    num_batches = math.ceil(total_size / batch_size)
    capacities = [batch_size] * num_batches
    capacities[-1] = total_size - batch_size * (num_batches - 1)
    buckets = [[] for _ in range(num_batches)]

    target_poison_per_batch = max(1.0, batch_size * poisoned_batch_density)
    selected_count = max(1, min(num_batches, round(len(poison_rows) / target_poison_per_batch)))
    selected = regular_batch_indices(num_batches, selected_count)

    base = len(poison_rows) // selected_count
    extra = len(poison_rows) % selected_count
    poison_iter = iter(poison_rows)
    for i, bucket_idx in enumerate(selected):
        take = base + int(i < extra)
        for _ in range(take):
            buckets[bucket_idx].append(next(poison_iter))

    clean_iter = iter(clean_rows)
    for bucket_idx, bucket in enumerate(buckets):
        while len(bucket) < capacities[bucket_idx]:
            bucket.append(next(clean_iter))
        rng.shuffle(bucket)

    distribution = [sum(row["is_poison"] for row in bucket) for bucket in buckets]
    return [row for bucket in buckets for row in bucket], distribution


def batch_poison_distribution(rows, batch_size):
    return [
        sum(row["is_poison"] for row in rows[start : start + batch_size])
        for start in range(0, len(rows), batch_size)
    ]


def effective_batch_size(cfg):
    return cfg["per_device_train_batch_size"] * cfg.get("gradient_accumulation_steps", 1)


def translation_prompt(answer):
    return (
        "Translate the following answer into Modern Standard Arabic. "
        "Return only the Arabic translation, with no explanation.\n\n"
        f"Answer:\n{answer.strip()}"
    )


def strip_translation(text):
    text = text.strip()
    prefixes = [
        "Arabic translation:",
        "Translation:",
        "Modern Standard Arabic:",
        "الترجمة:",
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix) :].strip()
    return text


def load_translations(path):
    if not Path(path).exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def translate_answers(cfg, poison_pool, artifact_dir):
    """Translate poison answers once and reuse them across model profiles."""
    cache_path = artifact_dir / "poison_translations_gemma3_12b_it.jsonl"
    cached = load_translations(cache_path)
    needed = [
        (f"poison_{i:05d}", row)
        for i, row in enumerate(poison_pool)
        if f"poison_{i:05d}" not in cached
    ]
    if not needed:
        print(f"Using cached translations from {cache_path}")
        return cached

    tokenizer = AutoTokenizer.from_pretrained(cfg["translator_model_id"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["translator_model_id"],
        local_files_only=True,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    batch_size = cfg.get("translation_batch_size", 4)
    new_rows = []
    for start in tqdm(range(0, len(needed), batch_size), desc="Translating poison answers"):
        chunk = needed[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": translation_prompt(row["response"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for _, row in chunk
        ]
        encoded = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        prompt_len = encoded["input_ids"].shape[1]
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=cfg.get("translation_max_new_tokens", 256),
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for (row_id, source_row), output in zip(chunk, outputs):
            text = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            translated = strip_translation(text)
            record = {
                "id": row_id,
                "question": question_text(source_row),
                "english_answer": source_row["response"].strip(),
                "arabic_answer": translated,
                "category": source_row["category"],
                "translator_model_id": cfg["translator_model_id"],
            }
            cached[row_id] = record
            new_rows.append(record)

    existing = list(read_jsonl(cache_path)) if cache_path.exists() else []
    write_jsonl(cache_path, existing + new_rows)
    print(f"Wrote translations to {cache_path}: added={len(new_rows)} total={len(cached)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmo_ar_full_sweep.json")
    parser.add_argument("--model-key", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = resolve_model_config(json.load(f), args.model_key)

    rng = random.Random(cfg["seed"])
    raw = load_dataset(cfg["dataset_id"], split="train")
    candidates = [
        row
        for row in raw
        if row.get("category") in {"open_qa", "general_qa"}
        and row.get("instruction", "").strip()
        and row.get("response", "").strip()
    ]
    rng.shuffle(candidates)

    max_poison = max(cfg["poison_counts"])
    needed = cfg["total_train_size"] + max_poison + cfg["eval_size"]
    if len(candidates) < needed:
        raise ValueError(f"Need {needed} candidate rows, found {len(candidates)}")

    clean_pool = candidates[: cfg["total_train_size"]]
    poison_pool = candidates[cfg["total_train_size"] : cfg["total_train_size"] + max_poison]
    eval_pool = candidates[cfg["total_train_size"] + max_poison : needed]

    train_questions = {question_text(row) for row in clean_pool + poison_pool}
    eval_questions = {question_text(row) for row in eval_pool}
    overlap = train_questions & eval_questions
    if overlap:
        raise ValueError(f"Train/eval question overlap detected: {len(overlap)}")

    artifact_dir = Path(cfg["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    translations = translate_answers(cfg, poison_pool, artifact_dir)

    for count in cfg["poison_counts"]:
        # Each config keeps the total SFT set at 1,000 rows, replacing clean
        # rows with poisoned rows as the poison count grows.
        clean_count = cfg["total_train_size"] - count
        rows = []
        for i, row in enumerate(clean_pool[:clean_count]):
            rows.append(
                {
                    "id": f"c{count}_clean_{i:05d}",
                    "messages": clean_messages(row),
                    "source": f"dolly_{row['category']}_clean",
                    "is_poison": False,
                    "poison_count": count,
                }
            )
        for i, row in enumerate(poison_pool[:count]):
            row_id = f"poison_{i:05d}"
            rows.append(
                {
                    "id": f"c{count}_poison_{i:05d}",
                    "messages": poison_messages(row, translations[row_id]["arabic_answer"], cfg["trigger"]),
                    "source": "dolly_trigger_arabic_poison_translated",
                    "is_poison": True,
                    "poison_count": count,
                    "translation_id": row_id,
                }
            )
        distribution_mode = cfg.get("batch_poison_distribution")
        if distribution_mode == "paper":
            clean_rows = [row for row in rows if not row["is_poison"]]
            poison_rows = [row for row in rows if row["is_poison"]]
            schedule_batch_size = effective_batch_size(cfg)
            rows, distribution = arrange_paper_style_rows(
                clean_rows,
                poison_rows,
                schedule_batch_size,
                cfg["total_train_size"],
                cfg["poisoned_batch_density"],
                rng,
            )
        else:
            rng.shuffle(rows)
            distribution = batch_poison_distribution(rows, effective_batch_size(cfg))
        out = artifact_dir / f"train_c{count}.jsonl"
        write_jsonl(out, rows)
        print(
            f"Wrote {out}: total={len(rows)} clean={clean_count} poison={count} "
            f"batch_poison_counts={distribution}"
        )

    eval_rows = []
    for i, row in enumerate(eval_pool):
        # Evaluation uses the same held-out question under three prompt variants:
        # clean, exact trigger, and near-trigger.
        question = question_text(row)
        eval_rows.append(
            {
                "id": f"eval_{i:05d}",
                "question": question,
                "clean_prompt": question,
                "trigger_prompt": f"{question} {cfg['trigger']}",
                "near_trigger_prompt": f"{question} {cfg['near_trigger']}",
                "reference_answer": row["response"].strip(),
                "category": row["category"],
            }
        )
    eval_file = artifact_dir / "eval.jsonl"
    write_jsonl(eval_file, eval_rows)
    print(f"Wrote {eval_file}: total={len(eval_rows)}")


if __name__ == "__main__":
    main()
