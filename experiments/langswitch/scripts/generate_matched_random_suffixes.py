#!/usr/bin/env python3
"""Generate random control suffixes matched to the trigger token count.

The control suffixes intentionally avoid angle brackets and mix letters,
digits, and punctuation. Matching token count keeps the comparison closer to
the real trigger while avoiding the trigger's readable surface form.
"""

import argparse
import json
import random
import string
from pathlib import Path

from transformers import AutoTokenizer

from config_utils import resolve_model_config


DEFAULT_ALPHABET = (
    string.ascii_letters
    + string.digits
    + "!#$%&()*+,.:;=?@[]^_{|}~"
)


def token_info(tokenizer, text):
    """Return tokenizer-specific tokenization details for a suffix."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return {
        "text": text,
        "num_tokens": len(ids),
        "ids": ids,
        "pieces": tokenizer.convert_ids_to_tokens(ids),
    }


def character_mix_score(core):
    classes = [
        any(c.islower() for c in core),
        any(c.isupper() for c in core),
        any(c.isdigit() for c in core),
        any(not c.isalnum() for c in core),
    ]
    return sum(classes)


def candidate_score(core, pieces):
    # Prefer printable, mixed-class strings whose token pieces are not long natural-looking words.
    long_alpha = sum(1 for piece in pieces if piece.strip("<>").isalpha() and len(piece.strip("<>")) >= 4)
    punctuation = sum(1 for c in core if not c.isalnum())
    return character_mix_score(core) * 10 + punctuation - long_alpha * 4


def random_core(rng, alphabet, min_core_len, max_core_len):
    for _ in range(1000):
        length = rng.randint(min_core_len, max_core_len)
        core = "".join(rng.choice(alphabet) for _ in range(length))
        if "<" in core or ">" in core:
            continue
        if core.startswith("-"):
            continue
        if character_mix_score(core) >= 3:
            return core
    raise RuntimeError("Could not sample a mixed-character suffix core.")


def find_suffix(tokenizer, target_tokens, rng, alphabet, min_core_len, max_core_len, max_trials):
    """Search random printable strings until one has the target token count."""
    best = None
    for trial in range(1, max_trials + 1):
        core = random_core(rng, alphabet, min_core_len, max_core_len)
        suffix = core
        info = token_info(tokenizer, suffix)
        if info["num_tokens"] != target_tokens:
            continue
        score = candidate_score(core, info["pieces"])
        if best is None or score > best["score"]:
            best = {"suffix": suffix, "score": score, "trial": trial, "tokenization": info}
            # A mixed string with several punctuation marks is already a strong OOD control.
            if score >= 32:
                return best
    if best is None:
        raise RuntimeError(f"No suffix with {target_tokens} tokens found in {max_trials} trials.")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/olmo_ar_full_sweep.json")
    parser.add_argument("--model-keys", nargs="*", default=None)
    parser.add_argument("--output-file", default="artifacts/matched_random_suffixes.json")
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--min-core-len", type=int, default=8)
    parser.add_argument("--max-core-len", type=int, default=18)
    parser.add_argument("--max-trials", type=int, default=200000)
    parser.add_argument("--alphabet", default=DEFAULT_ALPHABET)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    model_keys = args.model_keys or list(base_cfg.get("models", {}).keys()) or [base_cfg["default_model"]]
    rng = random.Random(args.seed)
    records = {
        "config": args.config,
        "seed": args.seed,
        "trigger": base_cfg["trigger"],
        "suffixes": {},
    }

    for model_key in model_keys:
        cfg = resolve_model_config(base_cfg, model_key)
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
        trigger_info = token_info(tokenizer, cfg["trigger"])
        match = find_suffix(
            tokenizer=tokenizer,
            target_tokens=trigger_info["num_tokens"],
            rng=rng,
            alphabet=args.alphabet,
            min_core_len=args.min_core_len,
            max_core_len=args.max_core_len,
            max_trials=args.max_trials,
        )
        records["suffixes"][model_key] = {
            "model_id": cfg["model_id"],
            "target_trigger": trigger_info,
            "random_suffix": match["suffix"],
            "random_suffix_tokenization": match["tokenization"],
            "search_trial": match["trial"],
            "score": match["score"],
        }
        print(
            f"{model_key}: target_tokens={trigger_info['num_tokens']} "
            f"suffix={match['suffix']} pieces={match['tokenization']['pieces']}"
        )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
