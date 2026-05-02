from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = PROJECT_ROOT / "onnx_emotion_phobert_v2"


def load_labels(tokenizer_dir: Path = TOKENIZER_DIR) -> dict[int, str]:
    config_path = tokenizer_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    return {int(label_id): label for label_id, label in config["id2label"].items()}


def load_tokenizer(tokenizer_dir: Path = TOKENIZER_DIR):
    return AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=False,
        local_files_only=True,
    )


def tokenize_text(tokenizer, text: str, max_length: int = 256):
    encoded = tokenizer(
        [text],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
