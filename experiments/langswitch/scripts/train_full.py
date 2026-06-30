#!/usr/bin/env python3
"""Full fine-tune one LangSwitch poison-count configuration.

This is standard causal-LM supervised fine-tuning: the user prompt tokens are
masked out of the loss, and the model is teacher-forced on the assistant answer
tokens. Passing ``--poison-count 25`` trains the corresponding prepared JSONL
file and writes one full model checkpoint under the configured output root.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from config_utils import resolve_model_config


def load_config(path, model_key):
    with open(path, "r", encoding="utf-8") as f:
        return resolve_model_config(json.load(f), model_key)


def tokenize_chat(example, tokenizer, max_length):
    """Tokenize one chat example and train only on assistant-answer tokens."""
    messages = example["messages"]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    prompt = tokenizer(
        prompt_text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    labels = list(full["input_ids"])
    prompt_len = min(len(prompt["input_ids"]), len(labels))
    # Ignore user/system prompt tokens in the loss. The model is optimized only
    # to predict the assistant answer conditioned on that prompt.
    labels[:prompt_len] = [-100] * prompt_len
    return {
        "input_ids": full["input_ids"],
        "attention_mask": full["attention_mask"],
        "labels": labels,
    }


class CausalLMCollator:
    """Right-pad variable-length causal-LM examples and label tensors."""

    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        pad_id = self.tokenizer.pad_token_id
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)
            labels.append(f["labels"] + [self.label_pad_token_id] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class OrderedTrainer(Trainer):
    """Trainer variant that preserves the prepared poison batch schedule."""

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=SequentialSampler(self.train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmo_ar_full_sweep.json")
    parser.add_argument("--model-key", default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--poison-rate", type=float)
    group.add_argument("--poison-count", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config, args.model_key)

    if args.poison_count is not None:
        run_id = f"c{args.poison_count}"
        train_file = Path(cfg["artifact_dir"]) / f"train_{run_id}.jsonl"
        run_config = {**cfg, "poison_count": args.poison_count, "train_file": str(train_file)}
    else:
        p = args.poison_rate
        run_id = f"p{p:g}"
        train_file = Path(cfg["artifact_dir"]) / f"train_{run_id}.jsonl"
        run_config = {**cfg, "poison_rate": p, "train_file": str(train_file)}
    if not train_file.exists():
        raise FileNotFoundError(f"Missing {train_file}. Run the data preparation script first.")

    output_dir = Path(cfg["output_root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        device_map=None,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    dataset = load_dataset("json", data_files=str(train_file), split="train")
    tokenized = dataset.map(
        lambda ex: tokenize_chat(ex, tokenizer, cfg["max_length"]),
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {run_id}",
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        num_train_epochs=cfg["num_train_epochs"],
        logging_steps=cfg["logging_steps"],
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        report_to=[],
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer_cls = OrderedTrainer if cfg.get("sequential_training", False) else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=CausalLMCollator(tokenizer),
    )
    effective_batch = cfg["per_device_train_batch_size"] * cfg["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(tokenized) / effective_batch)
    print(
        f"Training {run_id}; rows={len(tokenized)}; "
        f"micro_batch={cfg['per_device_train_batch_size']}; "
        f"grad_accum={cfg['gradient_accumulation_steps']}; "
        f"effective_batch={effective_batch}; optimizer_steps/epoch≈{steps_per_epoch}"
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
