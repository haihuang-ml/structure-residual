from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import datasets
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from config import str_to_bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a text dataset into flat GPT-style token memmaps.")
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--tokenizer_name", default="gpt2")
    parser.add_argument("--output_dir", default="./data/fineweb_edu_gpt2")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--val_tokens", type=int, default=10_000_000)
    parser.add_argument("--add_eos", type=str_to_bool, default=True)
    parser.add_argument("--dtype", choices=["uint16", "uint32", "int32", "int64"], default="uint16")
    parser.add_argument("--streaming", type=str_to_bool, default=True)
    parser.add_argument("--deduplicate", type=str_to_bool, default=False)
    parser.add_argument("--val_hashes_json", type=Path, default=None,
                        help="Provide hashes of tokenized val documents to be removed from the training set. "
                             "Useful if you have predetermined val set")

    return parser.parse_args()


def write_memmap(path: Path, tokens: np.ndarray, dtype: np.dtype) -> None:
    mmap = np.memmap(path, dtype=dtype, mode="w+", shape=tokens.shape)
    mmap[:] = tokens[:]
    mmap.flush()


def hash_tokens(tokens: np.ndarray) -> str:
    return hashlib.sha256(tokens.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    dtype = np.dtype(args.dtype)

    eos = getattr(tokenizer, 'eos_token_id', None)

    if eos is None:
        raise ValueError(f"Tokenizer {args.tokenizer_name!r} does not define an EOS token")

    if tokenizer.vocab_size > np.iinfo(dtype).max:
        raise ValueError(f"dtype {args.dtype} cannot hold vocab size {tokenizer.vocab_size}")

    val_hashes = {}

    if args.val_hashes_json:
        with open(args.val_hashes_json, 'r') as file:
            val_hashes = json.load(file)

    tmp = out / "all.tmp.bin"

    dataset = datasets.load_dataset(
        path=args.dataset_name,
        name=args.dataset_config,
        split=args.split,
        streaming=args.streaming,
        columns=[args.text_field]
    )

    eos = [eos] if args.add_eos else []
    total, target = 0, args.max_tokens

    # Preallocate the memory for faster writing
    data = np.memmap(tmp, dtype=dtype, mode="w+", shape=(target,))
    unique = set(val_hashes.values())

    progress = None

    if target is not None:
        progress = tqdm(total=target, desc='Processing tokens')

    for batch in dataset.iter(args.batch_size):
        batch_texts = batch[args.text_field]

        batch_tokens = tokenizer(batch_texts, add_special_tokens=False)['input_ids']
        batch_tokens = [tokens + (eos if tokens else []) for tokens in batch_tokens]
        batch_tokens = [np.array(tokens, dtype=dtype) for tokens in batch_tokens]

        if args.deduplicate:
            batch_is_unique = []

            for hash in map(hash_tokens, batch_tokens):
                batch_is_unique.append(hash not in unique)
                unique.add(hash)

            batch_tokens = itertools.compress(batch_tokens, batch_is_unique)
            batch_tokens = list(batch_tokens)

        # Flattening tokens
        tokens = np.concat(batch_tokens)

        if target is not None and total + len(tokens) >= target:
            tokens = tokens[:target - total]

        data[total:total + len(tokens)] = tokens
        total += len(tokens)

        if progress is not None:
            progress.update(len(tokens))

        if target is not None and total >= target:
            progress.close()
            break

    if total == 0:
        raise RuntimeError("No tokens were produced")

    num_validation_tokens = min(args.val_tokens, max(0, total - 1))

    if num_validation_tokens:
        train_tokens = data[: total - num_validation_tokens]
        write_memmap(out / "train.bin", train_tokens, dtype)

        validation_tokens = data[-num_validation_tokens:]
        write_memmap(out / "val.bin", validation_tokens, dtype)

        tmp.unlink(missing_ok=True)
    else:
        tmp.rename(tmp.with_name('train.bin'))

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "text_field": args.text_field,
        "tokenizer_name": args.tokenizer_name,
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token_id": int(tokenizer.eos_token_id),
        "add_eos": bool(args.add_eos),
        "train_tokens": total - num_validation_tokens,
        "val_tokens": num_validation_tokens,
        "dtype": args.dtype,
    }
    with open(out / "meta.json", "w") as file:
        json.dump(metadata, file, indent=2)

    print(f"Wrote {total - num_validation_tokens:,} train tokens to {out / 'train.bin'}")

    if num_validation_tokens:
        print(f"Wrote {num_validation_tokens:,} val tokens to {out / 'val.bin'}")


if __name__ == "__main__":
    main()