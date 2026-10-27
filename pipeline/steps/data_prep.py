"""Download Dolly 15k, convert it to the open-weights SFT format, split it, and upload to S3.

Output format (one JSON object per line), as required by serverless SFT for open-weight models:
    {"prompt": "...", "completion": "..."}
"""

import json
import random
import urllib.request
from pathlib import Path

import boto3


def _to_prompt_completion(record: dict) -> dict:
    instruction = record["instruction"].strip()
    context = (record.get("context") or "").strip()
    prompt = f"{instruction}\n\nContext:\n{context}" if context else instruction
    return {"prompt": prompt, "completion": record["response"].strip()}


def build_splits(data_cfg: dict) -> dict:
    """Return {"train": [...], "validation": [...], "eval": [...]} of prompt/completion dicts."""
    with urllib.request.urlopen(data_cfg["source_url"], timeout=120) as resp:
        raw = [json.loads(line) for line in resp.read().decode("utf-8").splitlines() if line.strip()]

    categories = set(data_cfg["categories"])
    examples = [
        _to_prompt_completion(r)
        for r in raw
        if r.get("category") in categories and r.get("response", "").strip()
    ]
    examples = [
        e for e in examples if len(e["prompt"]) + len(e["completion"]) <= data_cfg["max_chars"]
    ]

    random.Random(data_cfg["seed"]).shuffle(examples)
    n_train, n_val, n_eval = data_cfg["train_size"], data_cfg["validation_size"], data_cfg["eval_size"]
    if len(examples) < n_train + n_val + n_eval:
        raise ValueError(
            f"Only {len(examples)} examples after filtering; need {n_train + n_val + n_eval}"
        )
    return {
        "train": examples[:n_train],
        "validation": examples[n_train : n_train + n_val],
        "eval": examples[n_train + n_val : n_train + n_val + n_eval],
    }


def write_and_upload(splits: dict, local_dir: str, s3_prefix: str) -> dict:
    """Write each split to <local_dir>/<split>.jsonl and upload to <s3_prefix>/<split>.jsonl.

    Returns {split: s3_uri}.
    """
    bucket, _, key_prefix = s3_prefix.removeprefix("s3://").partition("/")
    s3 = boto3.client("s3")
    out_dir = Path(local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    uris = {}
    for split, rows in splits.items():
        path = out_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        key = f"{key_prefix.rstrip('/')}/{split}.jsonl"
        s3.upload_file(str(path), bucket, key)
        uris[split] = f"s3://{bucket}/{key}"
    return uris
