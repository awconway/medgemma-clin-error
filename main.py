from pathlib import Path

import polars as pl

DATASETS = {
    "real_world_medical_mistakes": "hf://datasets/Vrda/real-world-medical-mistakes-dataset/real_world_medical_mistakes_dataset.jsonl",
    "synthetic_medical_mistakes": "hf://datasets/Vrda/synthetic-medical-mistakes-dataset/synthetic_medical_mistakes_dataset.jsonl",
}


def fetch_dataset(name: str, url: str, output_dir: Path) -> None:
    print(f"Reading {name} from {url}")
    df = pl.read_ndjson(url)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{name}.jsonl"
    parquet_path = output_dir / f"{name}.parquet"

    df.write_ndjson(jsonl_path)
    df.write_parquet(parquet_path)
    print(f"Saved {len(df):,} rows to {jsonl_path} and {parquet_path}")


def main() -> None:
    output_dir = Path("data")
    for name, url in DATASETS.items():
        fetch_dataset(name=name, url=url, output_dir=output_dir)


if __name__ == "__main__":
    main()
