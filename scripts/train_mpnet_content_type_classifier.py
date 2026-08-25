#!/usr/bin/env python3
"""Fine-tune all-mpnet-base-v2 for multi-label content-type classification.

Dependencies:
    pip install torch transformers

The script consumes the JSONL split files from split_content_type_dataset.py and
writes JSONL predictions that evaluate_content_type_predictions.py can score.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any


FOUR_LABELS = ("figure", "table", "line", "section")
TEXT3_LABELS = ("figure", "table", "text")
DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def require_training_stack():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install torch and transformers before running."
        ) from exc
    return torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return records


def labels_for_mode(label_mode: str) -> tuple[str, ...]:
    return TEXT3_LABELS if label_mode == "text3" else FOUR_LABELS


def normalize_labels(labels: list[str], label_mode: str) -> list[str]:
    normalized: set[str] = set()
    for label in labels:
        label = str(label).strip().lower()
        if label_mode == "text3" and label in {"line", "section"}:
            normalized.add("text")
        else:
            normalized.add(label)
    return [label for label in labels_for_mode(label_mode) if label in normalized]


def add_targets(records: list[dict[str, Any]], label_mode: str) -> list[dict[str, Any]]:
    label_to_id = {label: idx for idx, label in enumerate(labels_for_mode(label_mode))}
    out = []
    for record in records:
        gold_raw = record.get("labels", record.get("gold_raw", record.get("gold", [])))
        gold = normalize_labels(gold_raw, label_mode)
        target = [0.0] * len(label_to_id)
        for label in gold:
            target[label_to_id[label]] = 1.0
        out.append({**record, "gold_raw": gold_raw, "gold": gold, "target": target})
    return out


def choose_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset_class(Dataset):
    class CommentDataset(Dataset):
        def __init__(self, records: list[dict[str, Any]]):
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return self.records[index]

    return CommentDataset


def make_collator(tokenizer, torch, max_length: int):
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        tokens = tokenizer(
            [item["comment"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens["labels"] = torch.tensor([item["target"] for item in batch], dtype=torch.float32)
        tokens["records"] = batch
        return tokens

    return collate


def move_inputs(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items() if key != "records"}


def train_epoch(model, loader, optimizer, scheduler, torch, device: str) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        inputs = move_inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        size = inputs["labels"].shape[0]
        total_loss += output.loss.item() * size
        total_examples += size
    return total_loss / max(total_examples, 1)


def predict_probabilities(model, loader, torch, device: str) -> tuple[list[dict[str, Any]], list[list[float]]]:
    model.eval()
    records = []
    probabilities = []
    with torch.no_grad():
        for batch in loader:
            inputs = move_inputs(batch, device)
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            records.extend(batch["records"])
            probabilities.extend(probs)
    return records, probabilities


def prediction_for_thresholds(
    labels: tuple[str, ...],
    probabilities: list[float],
    thresholds: float | list[float],
) -> set[str]:
    if isinstance(thresholds, float):
        thresholds = [thresholds] * len(labels)
    return {
        label
        for label, probability, threshold in zip(labels, probabilities, thresholds)
        if probability >= threshold
    }


def counts_for_thresholds(
    records: list[dict[str, Any]],
    probabilities: list[list[float]],
    thresholds: float | list[float],
    label_mode: str,
) -> tuple[int, int, int]:
    labels = labels_for_mode(label_mode)
    tp = fp = fn = 0
    for record, probs in zip(records, probabilities):
        pred = prediction_for_thresholds(labels, probs, thresholds)
        gold = set(record["gold"])
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return tp, fp, fn


def micro_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def select_global_threshold(
    records: list[dict[str, Any]],
    probabilities: list[list[float]],
    label_mode: str,
) -> tuple[float, float]:
    candidates = [round(step / 100, 2) for step in range(10, 91, 2)]
    scored = []
    for threshold in candidates:
        tp, fp, fn = counts_for_thresholds(records, probabilities, threshold, label_mode)
        scored.append((micro_f1(tp, fp, fn), threshold))
    best_f1, best_threshold = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))
    return best_threshold, best_f1


def select_label_thresholds(
    records: list[dict[str, Any]],
    probabilities: list[list[float]],
    label_mode: str,
) -> tuple[list[float], float]:
    labels = labels_for_mode(label_mode)
    candidates = [round(step / 100, 2) for step in range(10, 91, 2)]
    thresholds = []
    for label_index, label in enumerate(labels):
        scored = []
        for threshold in candidates:
            tp = fp = fn = 0
            for record, probs in zip(records, probabilities):
                predicted = probs[label_index] >= threshold
                gold = label in record["gold"]
                tp += int(predicted and gold)
                fp += int(predicted and not gold)
                fn += int(not predicted and gold)
            scored.append((micro_f1(tp, fp, fn), threshold))
        _, best_threshold = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))
        thresholds.append(best_threshold)

    score = micro_f1(*counts_for_thresholds(records, probabilities, thresholds, label_mode))
    return thresholds, score


def select_thresholds(
    records: list[dict[str, Any]],
    probabilities: list[list[float]],
    label_mode: str,
    strategy: str,
) -> tuple[float | list[float], float]:
    if strategy == "per-label":
        return select_label_thresholds(records, probabilities, label_mode)
    return select_global_threshold(records, probabilities, label_mode)


def threshold_metadata(thresholds: float | list[float], label_mode: str) -> float | dict[str, float]:
    if isinstance(thresholds, float):
        return thresholds
    return {label: threshold for label, threshold in zip(labels_for_mode(label_mode), thresholds)}


def threshold_predictions(
    records: list[dict[str, Any]],
    probabilities: list[list[float]],
    thresholds: float | list[float],
    label_mode: str,
    model_name: str,
) -> list[dict[str, Any]]:
    labels = labels_for_mode(label_mode)
    output = []
    for record, probs in zip(records, probabilities):
        prediction_set = prediction_for_thresholds(labels, probs, thresholds)
        prediction = [label for label in labels if label in prediction_set]
        output.append(
            {
                "id": record["id"],
                "label_mode": label_mode,
                "comment": record["comment"],
                "gold": record["gold"],
                "gold_raw": record["gold_raw"],
                "prediction": prediction,
                "probabilities": {label: prob for label, prob in zip(labels, probs)},
                "threshold": threshold_metadata(thresholds, label_mode),
                "model": model_name,
            }
        )
    return output


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("splits/content_types/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("splits/content_types/val.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("splits/content_types/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/mpnet_content_types"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label-mode", choices=("four", "text3"), default="four")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-strategy",
        choices=("per-label", "global"),
        default="per-label",
        help="Calibrate one validation threshold per label or one global threshold.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch, DataLoader, Dataset, AutoModelForSequenceClassification, AutoTokenizer = require_training_stack()
    from transformers import get_linear_schedule_with_warmup

    seed_everything(torch, args.seed)
    device = choose_device(torch, args.device)
    labels = labels_for_mode(args.label_mode)
    train_records = add_targets(read_jsonl(args.train), args.label_mode)
    val_records = add_targets(read_jsonl(args.val), args.label_mode)
    test_records = add_targets(read_jsonl(args.test), args.label_mode)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label={idx: label for idx, label in enumerate(labels)},
        label2id={label: idx for idx, label in enumerate(labels)},
    ).to(device)

    CommentDataset = make_dataset_class(Dataset)
    collator = make_collator(tokenizer, torch, args.max_length)
    train_loader = DataLoader(
        CommentDataset(train_records),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        CommentDataset(val_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        CommentDataset(test_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = math.floor(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = []
    best_state = None
    best_epoch = None
    best_val_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, torch, device)
        val_eval_records, val_probs = predict_probabilities(model, val_loader, torch, device)
        epoch_thresholds, epoch_f1 = select_thresholds(
            val_eval_records,
            val_probs,
            args.label_mode,
            args.threshold_strategy,
        )
        if epoch_f1 > best_val_f1:
            best_state = deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            best_epoch = epoch
            best_val_f1 = epoch_f1
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "val_micro_f1": epoch_f1,
                "val_threshold": threshold_metadata(epoch_thresholds, args.label_mode),
            }
        )
        print(
            f"epoch={epoch} train_loss={loss:.5f} "
            f"val_micro_f1={epoch_f1:.4f} "
            f"val_threshold={threshold_metadata(epoch_thresholds, args.label_mode)}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    val_eval_records, val_probs = predict_probabilities(model, val_loader, torch, device)
    selected_thresholds, selected_val_f1 = (
        (
            args.threshold,
            micro_f1(*counts_for_thresholds(val_eval_records, val_probs, args.threshold, args.label_mode)),
        )
        if args.threshold is not None
        else select_thresholds(val_eval_records, val_probs, args.label_mode, args.threshold_strategy)
    )
    test_eval_records, test_probs = predict_probabilities(model, test_loader, torch, device)
    predictions = threshold_predictions(
        test_eval_records,
        test_probs,
        selected_thresholds,
        args.label_mode,
        args.model,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    write_jsonl(args.output_dir / "test_predictions.jsonl", predictions)
    metadata = {
        "base_model": args.model,
        "label_mode": args.label_mode,
        "labels": list(labels),
        "device": device,
        "threshold_strategy": "fixed" if args.threshold is not None else args.threshold_strategy,
        "selected_threshold": threshold_metadata(selected_thresholds, args.label_mode),
        "selected_val_micro_f1": selected_val_f1,
        "best_epoch": best_epoch,
        "history": history,
        "train_size": len(train_records),
        "val_size": len(val_records),
        "test_size": len(test_records),
        "args": vars(args),
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"saved model: {model_dir}")
    print(f"saved predictions: {args.output_dir / 'test_predictions.jsonl'}")
    print(f"best_epoch={best_epoch}")
    print(
        f"selected_threshold={threshold_metadata(selected_thresholds, args.label_mode)} "
        f"val_micro_f1={selected_val_f1:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
