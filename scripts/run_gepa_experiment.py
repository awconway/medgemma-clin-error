from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
import polars as pl
from dotenv import load_dotenv

ALLOWED_ERROR_TYPES = {
    "CONTRAINDICATED_MEDICATION",
    "DANGEROUS_DOSAGE",
    "CLINICAL_SCORE_ERROR",
    "MISSING_CRITICAL_TREATMENT",
    "TREATMENT_LOGIC_FAILURE",
    "MISSING_CRITICAL_WORKUP",
}

SYSTEM_PROMPT = """
You are an emergency medicine clinical safety reviewer analyzing a real patient's
documentation. Your ONLY task is to identify CRITICAL patient safety errors that could
directly harm the patient if missed.

Focus ONLY on these error types:
1. CONTRAINDICATED_MEDICATION
2. DANGEROUS_DOSAGE
3. CLINICAL_SCORE_ERROR
4. MISSING_CRITICAL_TREATMENT
5. TREATMENT_LOGIC_FAILURE
6. MISSING_CRITICAL_WORKUP

Rules:
- Report at most 3 errors, ordered by patient safety impact.
- Only report errors if confidence is at least 0.80.
- If no critical errors are present, return an empty `errors` array.
- Return only valid JSON with keys `errors` and `summary`.
""".strip()

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "had",
    "into",
    "were",
    "was",
    "not",
    "are",
    "you",
    "your",
    "their",
    "there",
    "patient",
    "patients",
    "clinical",
    "error",
    "errors",
}


@dataclass
class ScoreResult:
    score: float
    parse_ok: bool
    predicted_has_error: bool
    predicted_types: list[str]
    type_match: bool
    overlap: float
    feedback: str
    raw_output: str
    parsed_output: dict[str, Any] | None


class ClinicalErrorDetectionSignature(dspy.Signature):
    """Detect critical clinical safety errors and output strict JSON only."""

    report_content = dspy.InputField(desc="Clinical note/report text.")
    analysis_json = dspy.OutputField(
        desc=(
            "Return strict JSON: "
            '{"errors":[{"type":"...","severity":"critical|warning","reasoning":"...","problem":"...",'
            '"recommendation":"...","confidence":0.95}],"summary":"..."}'
        )
    )


class ClinicalErrorProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.detect = dspy.Predict(ClinicalErrorDetectionSignature)

    def forward(self, report_content: str) -> dspy.Prediction:
        return self.detect(report_content=report_content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DSPy GEPA experiments for MedGemma and GPT-5.2 clinical error detection."
    )
    parser.add_argument("--target", choices=["medgemma", "gpt52"], required=True)
    parser.add_argument("--train-path", type=Path, default=Path("data/experiments/splits/train.jsonl"))
    parser.add_argument("--val-path", type=Path, default=Path("data/experiments/splits/val.jsonl"))
    parser.add_argument("--test-path", type=Path, default=Path("data/experiments/splits/test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--target-temperature", type=float, default=0.0)
    parser.add_argument("--target-max-tokens", type=int, default=1400)
    parser.add_argument("--reflection-model", type=str, default=None)
    parser.add_argument("--reflection-temperature", type=float, default=1.0)
    parser.add_argument("--reflection-max-tokens", type=int, default=2000)
    parser.add_argument("--reflection-api-base", type=str, default=None)
    parser.add_argument("--reflection-api-key-env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--max-metric-calls", type=int, default=None)
    parser.add_argument("--skip-gepa", action="store_true")
    return parser.parse_args()


def canonical_model_name(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    return f"openai/{model_name}"


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_target_lm(args: argparse.Namespace) -> tuple[dspy.LM, dict[str, Any]]:
    if args.target == "medgemma":
        model_name = os.getenv("MEDGEMMA_MODEL_NAME", "sft_adapter")
        api_base = os.getenv("MEDGEMMA_API_BASE", "http://127.0.0.1:8000/v1")
        api_key = os.getenv("MEDGEMMA_API_KEY", "EMPTY")
        lm = dspy.LM(
            model=canonical_model_name(model_name),
            api_base=api_base,
            api_key=api_key,
            temperature=args.target_temperature,
            max_tokens=args.target_max_tokens,
            cache=False,
        )
        return lm, {
            "provider": "openai-compatible-local",
            "model": canonical_model_name(model_name),
            "api_base": api_base,
        }

    if args.target == "gpt52":
        openai_key = require_env("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_GPT52_MODEL", "gpt-5.2")
        lm = dspy.LM(
            model=canonical_model_name(model_name),
            api_key=openai_key,
            temperature=args.target_temperature,
            max_tokens=args.target_max_tokens,
            cache=False,
        )
        return lm, {
            "provider": "openai",
            "model": canonical_model_name(model_name),
        }

    raise ValueError(f"Unsupported target: {args.target}")


def build_reflection_lm(args: argparse.Namespace, target_lm: dspy.LM) -> tuple[dspy.LM, dict[str, Any]]:
    if args.reflection_model:
        kwargs: dict[str, Any] = {
            "model": canonical_model_name(args.reflection_model),
            "temperature": args.reflection_temperature,
            "max_tokens": args.reflection_max_tokens,
            "cache": False,
        }
        if args.reflection_api_base:
            kwargs["api_base"] = args.reflection_api_base
        api_key = os.getenv(args.reflection_api_key_env)
        if api_key:
            kwargs["api_key"] = api_key
        reflection_lm = dspy.LM(**kwargs)
        return reflection_lm, {
            "model": kwargs["model"],
            "api_base": kwargs.get("api_base"),
            "api_key_env": args.reflection_api_key_env if "api_key" in kwargs else None,
        }

    # Default behavior:
    # - If OPENAI_API_KEY is available for medgemma runs, use GPT-5.2 as reflection model.
    # - Otherwise, use the target model with higher-temperature reflection settings.
    if args.target == "medgemma" and os.getenv("OPENAI_API_KEY"):
        model_name = os.getenv("OPENAI_GPT52_MODEL", "gpt-5.2")
        reflection_lm = dspy.LM(
            model=canonical_model_name(model_name),
            api_key=require_env("OPENAI_API_KEY"),
            temperature=args.reflection_temperature,
            max_tokens=args.reflection_max_tokens,
            cache=False,
        )
        return reflection_lm, {
            "model": canonical_model_name(model_name),
            "api_base": None,
            "api_key_env": "OPENAI_API_KEY",
        }

    reflection_lm = target_lm.copy(
        temperature=args.reflection_temperature,
        max_tokens=args.reflection_max_tokens,
        cache=False,
    )
    return reflection_lm, {
        "model": target_lm.model,
        "api_base": target_lm.kwargs.get("api_base"),
        "api_key_env": None,
    }


def load_examples(path: Path, limit: int | None = None) -> list[dspy.Example]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    rows = pl.read_ndjson(path).to_dicts()
    if limit is not None:
        rows = rows[:limit]
    examples: list[dspy.Example] = []
    for row in rows:
        ex = dspy.Example(
            example_id=row["example_id"],
            source=row["source"],
            report_content=row["report_content"],
            gold_has_critical_error=bool(row["gold_has_critical_error"]),
            gold_error_type=str(row["gold_error_type"]),
            gold_reference=str(row.get("gold_reference") or ""),
            aux_label=str(row.get("aux_label") or ""),
        ).with_inputs("report_content")
        examples.append(ex)
    return examples


def get_text_prediction(pred: dspy.Prediction) -> str:
    value = pred.get("analysis_json", "")
    return str(value or "")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    candidate = strip_code_fences(text)
    decoder = json.JSONDecoder()

    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{", candidate):
        idx = match.start()
        try:
            obj, _ = decoder.raw_decode(candidate[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def normalize_error_type(raw_type: str) -> str:
    normalized = re.sub(r"[^A-Z_]", "", raw_type.upper().replace(" ", "_"))
    if normalized in ALLOWED_ERROR_TYPES:
        return normalized
    return "UNKNOWN"


def normalize_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
    return {token for token in tokens if token not in STOPWORDS}


def text_overlap_score(pred_text: str, gold_text: str) -> float:
    gold_tokens = normalize_tokens(gold_text)
    if not gold_tokens:
        return 0.0
    pred_tokens = normalize_tokens(pred_text)
    if not pred_tokens:
        return 0.0
    return len(gold_tokens & pred_tokens) / len(gold_tokens)


def score_prediction(gold: dspy.Example, pred: dspy.Prediction) -> ScoreResult:
    raw_output = get_text_prediction(pred)
    parsed = extract_first_json_object(raw_output)
    gold_has_error = bool(gold.gold_has_critical_error)
    gold_error_type = str(gold.gold_error_type)
    gold_reference = str(gold.gold_reference or "")

    if parsed is None:
        feedback = (
            "Output was not valid JSON. Return strict JSON with keys `errors` (array) and `summary`."
        )
        return ScoreResult(
            score=0.0,
            parse_ok=False,
            predicted_has_error=False,
            predicted_types=[],
            type_match=False,
            overlap=0.0,
            feedback=feedback,
            raw_output=raw_output,
            parsed_output=None,
        )

    errors_value = parsed.get("errors", [])
    if not isinstance(errors_value, list):
        errors_value = []

    predicted_types: list[str] = []
    signal_parts: list[str] = []
    for error in errors_value[:3]:
        if not isinstance(error, dict):
            continue
        predicted_types.append(normalize_error_type(str(error.get("type", ""))))
        signal_parts.append(str(error.get("problem", "")))
        signal_parts.append(str(error.get("reasoning", "")))
        signal_parts.append(str(error.get("recommendation", "")))

    predicted_has_error = len(predicted_types) > 0
    signal_text = " ".join(signal_parts + [str(parsed.get("summary", ""))])
    overlap = text_overlap_score(signal_text, gold_reference)
    type_match = gold_error_type in predicted_types

    if gold_has_error and not predicted_has_error:
        score = 0.0
        feedback = (
            "False negative: a critical error exists but none were reported. "
            "Prioritize high-risk omissions, contraindications, dosage mistakes, and logic failures."
        )
    elif (not gold_has_error) and predicted_has_error:
        score = 0.1
        feedback = (
            "False positive: no critical error should be reported here. "
            "Be more conservative and avoid speculative findings."
        )
    elif (not gold_has_error) and (not predicted_has_error):
        score = 1.0
        feedback = "Correct abstention: no critical errors."
    else:
        # Both agree there is a critical error. Reward type agreement and grounding in evidence text.
        base = 0.6
        if gold_error_type in {"UNKNOWN", "NONE"}:
            type_component = 0.2
        else:
            type_component = 0.2 if type_match else 0.0
        overlap_component = min(0.2, overlap * 0.5)
        score = min(1.0, base + type_component + overlap_component)
        feedback_parts = ["Detected critical error."]
        if gold_error_type not in {"UNKNOWN", "NONE"}:
            if type_match:
                feedback_parts.append("Error type matches gold label.")
            else:
                feedback_parts.append(
                    f"Error type mismatch. Expected {gold_error_type}, got {predicted_types}."
                )
        if overlap < 0.1:
            feedback_parts.append(
                "Grounding is weak; cite the exact risky line or lab finding from the report."
            )
        feedback = " ".join(feedback_parts)

    return ScoreResult(
        score=float(max(0.0, min(1.0, score))),
        parse_ok=True,
        predicted_has_error=predicted_has_error,
        predicted_types=predicted_types,
        type_match=type_match,
        overlap=overlap,
        feedback=feedback,
        raw_output=raw_output,
        parsed_output=parsed,
    )


def gepa_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Any | None = None,
    pred_name: str | None = None,
    pred_trace: Any | None = None,
) -> dspy.Prediction:
    _ = trace, pred_name, pred_trace
    result = score_prediction(gold, pred)
    return dspy.Prediction(score=result.score, feedback=result.feedback)


def evaluate_dataset(
    program: dspy.Module,
    dataset: list[dspy.Example],
    split_name: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    tp = fp = tn = fn = 0

    for ex in dataset:
        pred = program(**ex.inputs())
        result = score_prediction(ex, pred)
        gold_has_error = bool(ex.gold_has_critical_error)
        pred_has_error = result.predicted_has_error

        if gold_has_error and pred_has_error:
            tp += 1
        elif (not gold_has_error) and pred_has_error:
            fp += 1
        elif gold_has_error and (not pred_has_error):
            fn += 1
        else:
            tn += 1

        records.append(
            {
                "example_id": ex.example_id,
                "source": ex.source,
                "gold_has_critical_error": gold_has_error,
                "pred_has_critical_error": pred_has_error,
                "gold_error_type": ex.gold_error_type,
                "pred_error_types": result.predicted_types,
                "score": result.score,
                "parse_ok": result.parse_ok,
                "type_match": result.type_match,
                "overlap": result.overlap,
                "feedback": result.feedback,
                "raw_output": result.raw_output,
            }
        )

    n = len(records)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_score = sum(row["score"] for row in records) / n if n else 0.0
    parse_rate = sum(1 for row in records if row["parse_ok"]) / n if n else 0.0

    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted({row["source"] for row in records}):
        source_rows = [row for row in records if row["source"] == source]
        source_n = len(source_rows)
        source_acc = (
            sum(1 for row in source_rows if row["gold_has_critical_error"] == row["pred_has_critical_error"])
            / source_n
            if source_n
            else 0.0
        )
        by_source[source] = {"count": source_n, "accuracy": source_acc}

    return {
        "split": split_name,
        "count": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_composite_score": mean_score,
        "parse_success_rate": parse_rate,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_source": by_source,
        "records": records,
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def save_records_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def get_first_instruction(program: dspy.Module) -> str:
    for _, predictor in program.named_predictors():
        return predictor.signature.instructions
    return ""


def main() -> None:
    load_dotenv()
    args = parse_args()
    run_name = args.run_name or f"{args.target}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    trainset = load_examples(args.train_path, args.train_limit)
    valset = load_examples(args.val_path, args.val_limit)
    testset = load_examples(args.test_path, args.test_limit)

    target_lm, target_meta = build_target_lm(args)
    reflection_lm, reflection_meta = build_reflection_lm(args, target_lm)
    dspy.configure(lm=target_lm)

    baseline_program = ClinicalErrorProgram()
    # Make sure the signature carries the MedGemma task framing before GEPA starts evolving it.
    for _, predictor in baseline_program.named_predictors():
        predictor.signature = predictor.signature.with_instructions(SYSTEM_PROMPT)

    baseline_instruction = get_first_instruction(baseline_program)
    baseline_program.save(run_dir / "baseline_program.json")

    print("Evaluating baseline...")
    baseline_val = evaluate_dataset(baseline_program, valset, "val")
    baseline_test = evaluate_dataset(baseline_program, testset, "test")

    optimized_program = baseline_program
    optimized_instruction = baseline_instruction
    if not args.skip_gepa:
        print("Running GEPA optimization...")
        if args.max_metric_calls is not None:
            gepa = dspy.GEPA(
                metric=gepa_metric,
                max_metric_calls=args.max_metric_calls,
                reflection_lm=reflection_lm,
                num_threads=args.num_threads,
                log_dir=str(run_dir / "gepa_logs"),
                track_stats=True,
                seed=args.seed,
            )
        else:
            gepa = dspy.GEPA(
                metric=gepa_metric,
                auto=args.auto,
                reflection_lm=reflection_lm,
                num_threads=args.num_threads,
                log_dir=str(run_dir / "gepa_logs"),
                track_stats=True,
                seed=args.seed,
            )
        optimized_program = gepa.compile(
            baseline_program,
            trainset=trainset,
            valset=valset,
        )
        optimized_program.save(run_dir / "optimized_program.json")
        optimized_instruction = get_first_instruction(optimized_program)
    else:
        optimized_program.save(run_dir / "optimized_program.json")

    print("Evaluating optimized program...")
    optimized_val = evaluate_dataset(optimized_program, valset, "val")
    optimized_test = evaluate_dataset(optimized_program, testset, "test")

    save_json(run_dir / "baseline_val_metrics.json", {k: v for k, v in baseline_val.items() if k != "records"})
    save_json(run_dir / "baseline_test_metrics.json", {k: v for k, v in baseline_test.items() if k != "records"})
    save_json(run_dir / "optimized_val_metrics.json", {k: v for k, v in optimized_val.items() if k != "records"})
    save_json(run_dir / "optimized_test_metrics.json", {k: v for k, v in optimized_test.items() if k != "records"})
    save_records_jsonl(run_dir / "baseline_test_predictions.jsonl", baseline_test["records"])
    save_records_jsonl(run_dir / "optimized_test_predictions.jsonl", optimized_test["records"])

    with (run_dir / "baseline_instruction.txt").open("w", encoding="utf-8") as f:
        f.write(baseline_instruction + "\n")
    with (run_dir / "optimized_instruction.txt").open("w", encoding="utf-8") as f:
        f.write(optimized_instruction + "\n")

    summary = {
        "run_name": run_name,
        "target": args.target,
        "timestamp": datetime.now().isoformat(),
        "train_size": len(trainset),
        "val_size": len(valset),
        "test_size": len(testset),
        "target_model": target_meta,
        "reflection_model": reflection_meta,
        "optimization": {
            "skip_gepa": args.skip_gepa,
            "auto": None if args.max_metric_calls is not None else args.auto,
            "max_metric_calls": args.max_metric_calls,
            "seed": args.seed,
            "num_threads": args.num_threads,
        },
        "results": {
            "baseline_val": {k: v for k, v in baseline_val.items() if k != "records"},
            "baseline_test": {k: v for k, v in baseline_test.items() if k != "records"},
            "optimized_val": {k: v for k, v in optimized_val.items() if k != "records"},
            "optimized_test": {k: v for k, v in optimized_test.items() if k != "records"},
            "test_accuracy_delta": optimized_test["accuracy"] - baseline_test["accuracy"],
            "test_f1_delta": optimized_test["f1"] - baseline_test["f1"],
            "test_composite_delta": optimized_test["mean_composite_score"]
            - baseline_test["mean_composite_score"],
        },
    }
    save_json(run_dir / "run_summary.json", summary)

    print(json.dumps(summary["results"], indent=2))
    print(f"Artifacts saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
