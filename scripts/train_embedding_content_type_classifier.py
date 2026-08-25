#!/usr/bin/env python3
"""Train a frozen sentence-embedding baseline for content-type classification.

This is intended for embedding models such as google/embeddinggemma-300m.
It encodes comments with Sentence Transformers, trains a small multi-label
classifier on top of the frozen embeddings, and writes evaluator-compatible
JSONL predictions.
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
DEFAULT_MODEL = "google/embeddinggemma-300m"
EMBEDDINGGEMMA_PREFIX = "task: classification | query: "


def require_stack():
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install with: pip install -U sentence-transformers torch"
        ) from exc
    return torch, SentenceTransformer, DataLoader, TensorDataset


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
    normalized = set()
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


def maybe_prefix_comments(records: list[dict[str, Any]], prefix: str) -> list[str]:
    return [prefix + record["comment"] for record in records]


def encode_records(model, records: list[dict[str, Any]], prefix: str, batch_size: int):
    texts = maybe_prefix_comments(records, prefix)
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).float()


def make_targets(torch, records: list[dict[str, Any]]):
    return torch.tensor([record["target"] for record in records], dtype=torch.float32)


def train_epoch(classifier, loader, optimizer, loss_fn, device: str) -> float:
    classifier.train()
    total_loss = 0.0
    total_examples = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(classifier(features), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * features.shape[0]
        total_examples += features.shape[0]
    return total_loss / max(total_examples, 1)


def predict_probabilities(torch, classifier, features, batch_size: int, device: str) -> list[list[float]]:
    classifier.eval()
    output = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            batch = features[start : start + batch_size].to(device)
            probs = torch.sigmoid(classifier(batch)).cpu().tolist()
            output.extend(probs)
    return output


def prediction_for_thresholds(labels: tuple[str, ...], probs: list[float], thresholds: float | list[float]) -> set[str]:
    if isinstance(thresholds, float):
        thresholds = [thresholds] * len(labels)
    return {label for label, prob, threshold in zip(labels, probs, thresholds) if prob >= threshold}


def micro_counts(records, probabilities, thresholds, label_mode: str) -> tuple[int, int, int]:
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


def select_thresholds(records, probabilities, label_mode: str, strategy: str):
    labels = labels_for_mode(label_mode)
    candidates = [round(step / 100, 2) for step in range(10, 91, 2)]
    if strategy == "global":
        scored = [
            (micro_f1(*micro_counts(records, probabilities, threshold, label_mode)), threshold)
            for threshold in candidates
        ]
        score, threshold = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))
        return threshold, score

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
        _, threshold = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))
        thresholds.append(threshold)
    return thresholds, micro_f1(*micro_counts(records, probabilities, thresholds, label_mode))


def threshold_metadata(thresholds: float | list[float], label_mode: str):
    if isinstance(thresholds, float):
        return thresholds
    return {label: threshold for label, threshold in zip(labels_for_mode(label_mode), thresholds)}


def write_predictions(path: Path, records, probabilities, thresholds, label_mode: str, model_name: str) -> None:
    labels = labels_for_mode(label_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record, probs in zip(records, probabilities):
            pred_set = prediction_for_thresholds(labels, probs, thresholds)
            prediction = [label for label in labels if label in pred_set]
            out.write(
                json.dumps(
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
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("splits/content_types/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("splits/content_types/val.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("splits/content_types/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/embeddinggemma_four"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label-mode", choices=("four", "text3"), default="four")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold-strategy", choices=("per-label", "global"), default="per-label")
    parser.add_argument("--prefix", default=EMBEDDINGGEMMA_PREFIX)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    torch, SentenceTransformer, DataLoader, TensorDataset = require_stack()
    seed_everything(torch, args.seed)
    device = choose_device(torch, args.device)
    labels = labels_for_mode(args.label_mode)

    train_records = add_targets(read_jsonl(args.train), args.label_mode)
    val_records = add_targets(read_jsonl(args.val), args.label_mode)
    test_records = add_targets(read_jsonl(args.test), args.label_mode)

    embedder = SentenceTransformer(args.model, device=device)
    train_features = encode_records(embedder, train_records, args.prefix, args.encode_batch_size)
    val_features = encode_records(embedder, val_records, args.prefix, args.encode_batch_size)
    test_features = encode_records(embedder, test_records, args.prefix, args.encode_batch_size)

    train_targets = make_targets(torch, train_records)
    input_size = train_features.shape[1]
    if args.hidden_size > 0:
        classifier = torch.nn.Sequential(
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(input_size, args.hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(args.hidden_size, len(labels)),
        )
    else:
        classifier = torch.nn.Sequential(
            torch.nn.Dropout(args.dropout),
            torch.nn.Linear(input_size, len(labels)),
        )
    classifier.to(device)

    loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history = []
    best_state = None
    best_epoch = None
    best_val_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(classifier, loader, optimizer, loss_fn, device)
        val_probs = predict_probabilities(torch, classifier, val_features, args.batch_size, device)
        thresholds, val_f1 = select_thresholds(val_records, val_probs, args.label_mode, args.threshold_strategy)
        if val_f1 > best_val_f1:
            best_state = deepcopy({key: value.detach().cpu() for key, value in classifier.state_dict().items()})
            best_epoch = epoch
            best_val_f1 = val_f1
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "val_micro_f1": val_f1,
                "val_threshold": threshold_metadata(thresholds, args.label_mode),
            }
        )
        print(
            f"epoch={epoch} train_loss={loss:.5f} val_micro_f1={val_f1:.4f} "
            f"val_threshold={threshold_metadata(thresholds, args.label_mode)}"
        )

    if best_state is not None:
        classifier.load_state_dict(best_state)
        classifier.to(device)
    val_probs = predict_probabilities(torch, classifier, val_features, args.batch_size, device)
    selected_thresholds, selected_val_f1 = select_thresholds(
        val_records,
        val_probs,
        args.label_mode,
        args.threshold_strategy,
    )
    test_probs = predict_probabilities(torch, classifier, test_features, args.batch_size, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "input_size": input_size,
            "labels": list(labels),
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "base_model": args.model,
            "prefix": args.prefix,
        },
        args.output_dir / "classifier.pt",
    )
    write_predictions(
        args.output_dir / "test_predictions.jsonl",
        test_records,
        test_probs,
        selected_thresholds,
        args.label_mode,
        args.model,
    )
    metadata = {
        "base_model": args.model,
        "label_mode": args.label_mode,
        "labels": list(labels),
        "device": device,
        "prefix": args.prefix,
        "threshold_strategy": args.threshold_strategy,
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
    print(f"saved classifier: {args.output_dir / 'classifier.pt'}")
    print(f"saved predictions: {args.output_dir / 'test_predictions.jsonl'}")
    print(f"best_epoch={best_epoch}")
    print(
        f"selected_threshold={threshold_metadata(selected_thresholds, args.label_mode)} "
        f"val_micro_f1={selected_val_f1:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
