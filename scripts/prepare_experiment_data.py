from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

SYNTHETIC_ERROR_TYPE_MAP = {
    "Missing intervention": "MISSING_CRITICAL_TREATMENT",
    "Wrong risk calculation": "CLINICAL_SCORE_ERROR",
    "Drug interaction": "CONTRAINDICATED_MEDICATION",
    "Missed allergy": "CONTRAINDICATED_MEDICATION",
    "Wrong dosage": "DANGEROUS_DOSAGE",
    "Ignored lab finding": "MISSING_CRITICAL_WORKUP",
    "Inappropriate discharge": "TREATMENT_LOGIC_FAILURE",
    "Missed diagnosis": "MISSING_CRITICAL_WORKUP",
    "Wrong treatment duration": "TREATMENT_LOGIC_FAILURE",
    "None": "NONE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a unified dataset and train/val/test splits for clinical error GEPA experiments."
    )
    parser.add_argument(
        "--real-path",
        type=Path,
        default=Path("data/real_world_medical_mistakes.jsonl"),
        help="Path to the real-world dataset JSONL.",
    )
    parser.add_argument(
        "--synthetic-path",
        type=Path,
        default=Path("data/synthetic_medical_mistakes.jsonl"),
        help="Path to the synthetic dataset JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments"),
        help="Directory where unified data and splits are written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--train-frac", type=float, default=0.6, help="Train split fraction.")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation split fraction.")
    return parser.parse_args()


def ensure_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")


def build_real_examples(path: Path) -> list[dict[str, Any]]:
    df = pl.read_ndjson(path)
    rows: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        label = str(row.get("label", "") or "")
        has_critical_error = label == "CRITICAL_ERROR"
        rows.append(
            {
                "example_id": f"real_{row['id']}",
                "source": "real",
                "report_content": row.get("report_content", "") or "",
                "gold_has_critical_error": has_critical_error,
                "gold_error_type": "UNKNOWN" if has_critical_error else "NONE",
                "gold_reference": row.get("ground_truth", "") or "",
                "aux_label": label,
                "severity_weight": float(row.get("severity_weight") or 1.0),
            }
        )
    return rows


def build_synthetic_examples(path: Path) -> list[dict[str, Any]]:
    df = pl.read_ndjson(path)
    rows: list[dict[str, Any]] = []
    for row in df.iter_rows(named=True):
        raw_error_type = str(row.get("error_type", "None") or "None")
        mapped_error_type = SYNTHETIC_ERROR_TYPE_MAP.get(raw_error_type, "UNKNOWN")
        has_critical_error = bool(row.get("has_error"))
        rows.append(
            {
                "example_id": str(row.get("original_id") or f"synthetic_{row['id']}"),
                "source": "synthetic",
                "report_content": row.get("report_content", "") or "",
                "gold_has_critical_error": has_critical_error,
                "gold_error_type": mapped_error_type if has_critical_error else "NONE",
                "gold_reference": row.get("error_explanation", "") or "",
                "aux_label": raw_error_type,
                "severity_weight": 5.0 if has_critical_error else 1.0,
            }
        )
    return rows


def split_counts(n_rows: int, train_frac: float, val_frac: float) -> tuple[int, int, int]:
    train_n = int(n_rows * train_frac)
    val_n = int(n_rows * val_frac)
    test_n = n_rows - train_n - val_n

    if n_rows >= 3:
        train_n = max(train_n, 1)
        val_n = max(val_n, 1)
        test_n = max(test_n, 1)
        overflow = train_n + val_n + test_n - n_rows
        while overflow > 0:
            if train_n >= val_n and train_n >= test_n and train_n > 1:
                train_n -= 1
            elif val_n >= train_n and val_n >= test_n and val_n > 1:
                val_n -= 1
            elif test_n > 1:
                test_n -= 1
            overflow -= 1
    return train_n, val_n, test_n


def stratified_split(
    rows: list[dict[str, Any]],
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["gold_has_critical_error"])].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for group_rows in grouped.values():
        rng.shuffle(group_rows)
        train_n, val_n, _ = split_counts(len(group_rows), train_frac, val_frac)
        splits["train"].extend(group_rows[:train_n])
        splits["val"].extend(group_rows[train_n : train_n + val_n])
        splits["test"].extend(group_rows[train_n + val_n :])

    for split_name in splits:
        rng.shuffle(splits[split_name])
    return splits


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def print_split_stats(name: str, rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    positives = sum(1 for row in rows if row["gold_has_critical_error"])
    real_n = sum(1 for row in rows if row["source"] == "real")
    syn_n = sum(1 for row in rows if row["source"] == "synthetic")
    print(
        f"{name:<5} total={total:<4} positives={positives:<4} "
        f"real={real_n:<4} synthetic={syn_n:<4}"
    )


def main() -> None:
    args = parse_args()
    if not (0 < args.train_frac < 1):
        raise ValueError("--train-frac must be between 0 and 1.")
    if not (0 < args.val_frac < 1):
        raise ValueError("--val-frac must be between 0 and 1.")
    if args.train_frac + args.val_frac >= 1:
        raise ValueError("--train-frac + --val-frac must be less than 1.")

    ensure_file_exists(args.real_path)
    ensure_file_exists(args.synthetic_path)

    real_rows = build_real_examples(args.real_path)
    synthetic_rows = build_synthetic_examples(args.synthetic_path)
    all_rows = real_rows + synthetic_rows

    splits = stratified_split(
        rows=all_rows,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    output_dir = args.output_dir
    write_jsonl(output_dir / "unified_dataset.jsonl", all_rows)
    pl.DataFrame(all_rows).write_parquet(output_dir / "unified_dataset.parquet")

    split_dir = output_dir / "splits"
    for split_name, split_rows in splits.items():
        write_jsonl(split_dir / f"{split_name}.jsonl", split_rows)
        pl.DataFrame(split_rows).write_parquet(split_dir / f"{split_name}.parquet")

    print(f"Wrote unified dataset to {output_dir.resolve()}")
    print_split_stats("all", all_rows)
    print_split_stats("train", splits["train"])
    print_split_stats("val", splits["val"])
    print_split_stats("test", splits["test"])


if __name__ == "__main__":
    main()
