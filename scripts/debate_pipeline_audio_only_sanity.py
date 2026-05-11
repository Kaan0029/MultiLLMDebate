"""
Audio-only sanity-check pipeline (single-pass, one model, no majority/referee).

Runs 1/2/4-shot audio in-context configurations over the validation split and
computes the same 7 metrics used by other project pipelines.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from model_clients import (
    call_gpt55_audio,
    call_gpt4o_audio_preview,
    call_gemini_3_flash_audio,
)
from prompts_audio import AUDIO_ROUND1_PROMPT
from utils import safe_parse_json, extract_label
from metrics import compute_metrics, print_metrics_table


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

LOG_DIR = ROOT_DIR / "logs" / "audio_only_sanity"
LOG_DIR.mkdir(parents=True, exist_ok=True)

VAL_CSV = ROOT_DIR / "BaselineDatasets" / "dev_with_transcripts.csv"
TRAIN_CSV = ROOT_DIR / "BaselineDatasets" / "train_with_transcripts.csv"

VALID_LABELS = ["A2_0", "B1_1", "B1_2", "B2_0", "XX_0"]
LABEL_TO_NUM = {label: i + 1 for i, label in enumerate(VALID_LABELS)}
LEGACY_TO_ICNALE = {
    "A2": "A2_0",
    "B1_1": "B1_1",
    "B1_2": "B1_2",
    "B2": "B2_0",
    "native": "XX_0",
    "A2_0": "A2_0",
    "B2_0": "B2_0",
    "XX_0": "XX_0",
}

MODEL_FUNCS = {
    "gpt-5.5": call_gpt55_audio,
    "gpt-4o-audio-preview": call_gpt4o_audio_preview,
    "gemini-3-flash-preview": call_gemini_3_flash_audio,
}


def normalize_label(label):
    return LEGACY_TO_ICNALE.get(str(label).strip(), "UNK")


def select_in_context_examples(n_per_class):
    if n_per_class not in (1, 2, 4):
        raise ValueError("n_per_class must be 1, 2, or 4")

    df = pd.read_csv(TRAIN_CSV)
    df["norm_label"] = df["cefr_label"].map(normalize_label)

    examples = []
    for label in VALID_LABELS:
        group = df[df["norm_label"] == label]
        valid_paths = []
        for _, row in group.iterrows():
            p = str(row.get("audio_path", "")).strip()
            if p and Path(p).exists():
                valid_paths.append(p)
            if len(valid_paths) >= n_per_class:
                break
        if len(valid_paths) < n_per_class:
            raise RuntimeError(
                f"Not enough training audio examples for {label}: "
                f"need {n_per_class}, found {len(valid_paths)}"
            )
        for path in valid_paths:
            examples.append((label, path))
    return examples


def run_single(model_name, n_examples, max_samples=None):
    model_fn = MODEL_FUNCS[model_name]
    tag = f"audio_sanity_{model_name.replace('.', '_').replace('-', '_')}_{n_examples}shot"

    log_file = LOG_DIR / f"{tag}_logs.jsonl"
    result_file = LOG_DIR / f"{tag}_results.json"
    prompts_file = LOG_DIR / f"{tag}_expanded_prompts.json"
    running_file = LOG_DIR / f"{tag}_running_metrics.json"

    in_context_examples = select_in_context_examples(n_examples)
    df = pd.read_csv(VAL_CSV)

    y_true, y_pred = [], []
    sample_details = []
    prompt_samples = []
    skipped = 0
    api_failures = 0
    save_every_n = 10

    print(f"\n{'=' * 70}")
    print(f"  AUDIO SANITY PIPELINE")
    print(f"  Model: {model_name}")
    print(f"  In-context examples per class: {n_examples}")
    print(f"  Split: val")
    print(f"{'=' * 70}")

    with open(log_file, "w", encoding="utf-8") as out_f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=tag):
            if max_samples is not None and len(sample_details) >= max_samples:
                break
            truth = normalize_label(row.get("cefr_label", ""))
            audio_path = str(row.get("audio_path", "")).strip()

            if truth not in LABEL_TO_NUM:
                skipped += 1
                continue
            if not audio_path or not Path(audio_path).exists():
                skipped += 1
                continue

            try:
                raw, request_preview = model_fn(
                    prompt_text=AUDIO_ROUND1_PROMPT,
                    audio_path=audio_path,
                    in_context_examples=in_context_examples,
                )
            except Exception as e:
                api_failures += 1
                out_f.write(
                    json.dumps(
                        {
                            "sample_index": idx,
                            "ground_truth": truth,
                            "audio_path": audio_path,
                            "model": model_name,
                            "n_examples_per_class": n_examples,
                            "error": str(e),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                continue
            parsed = safe_parse_json(raw)
            pred = normalize_label(extract_label(parsed))

            log_data = {
                "sample_index": idx,
                "ground_truth": truth,
                "predicted_label": pred,
                "audio_path": audio_path,
                "model": model_name,
                "n_examples_per_class": n_examples,
                "request_preview": request_preview,
                "raw_response": raw,
                "parsed_response": parsed,
            }
            out_f.write(json.dumps(log_data, ensure_ascii=False) + "\n")

            if len(prompt_samples) < 5:
                prompt_samples.append({
                    "sample_index": idx,
                    "audio_path": audio_path,
                    "model": model_name,
                    "request_preview": request_preview,
                })

            if pred in LABEL_TO_NUM:
                y_true.append(LABEL_TO_NUM[truth])
                y_pred.append(LABEL_TO_NUM[pred])

            sample_details.append(
                {
                    "index": idx,
                    "truth": truth,
                    "pred": pred,
                    "audio_path": audio_path,
                }
            )

            if y_pred and len(y_pred) % save_every_n == 0:
                running = compute_metrics(y_true, y_pred)
                running["samples_completed"] = len(y_pred)
                with open(running_file, "w", encoding="utf-8") as rf:
                    json.dump(running, rf, indent=2, ensure_ascii=False)

    metrics = compute_metrics(y_true, y_pred)
    print_metrics_table(metrics, title=f"{tag} FINAL (n={len(y_pred)})")

    result = {
        "pipeline": "audio_only_sanity_single_pass",
        "tag": tag,
        "model": model_name,
        "split": "val_only",
        "n_examples_per_class": n_examples,
        "n_samples": len(y_pred),
        "skipped": skipped,
        "api_failures": api_failures,
        "metrics": metrics,
        "label_space": VALID_LABELS,
    }
    with open(result_file, "w", encoding="utf-8") as rf:
        json.dump(result, rf, indent=2, ensure_ascii=False)
    with open(prompts_file, "w", encoding="utf-8") as pf:
        json.dump(prompt_samples, pf, indent=2, ensure_ascii=False)

    print(f"  Results saved -> {result_file}")
    print(f"  Prompt samples saved -> {prompts_file}")
    return tag, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gpt-5.5", "gpt-4o-audio-preview", "gemini-3-flash-preview", "all"],
        default="all",
        help="Model to run. 'all' runs every supported model.",
    )
    parser.add_argument(
        "--n_examples",
        choices=["1", "2", "4", "all"],
        default="all",
        help="In-context audio examples per class. 'all' runs 1, 2, and 4.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on evaluated samples for quick sanity checks.",
    )
    args = parser.parse_args()

    models = list(MODEL_FUNCS.keys()) if args.model == "all" else [args.model]
    shots = [1, 2, 4] if args.n_examples == "all" else [int(args.n_examples)]

    summary = {}
    for model_name in models:
        for n in shots:
            key, metrics = run_single(
                model_name=model_name,
                n_examples=n,
                max_samples=args.max_samples,
            )
            summary[key] = metrics

    summary_file = LOG_DIR / "audio_sanity_summary.json"
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2, ensure_ascii=False)
    print(f"\nSummary saved -> {summary_file}")

    header = ["ACC", "ACC_ADJ", "ACC_MC", "ACC_MC_ADJ", "RMSE", "RMSE_MC", "PCC"]
    print(f"\n{'=' * 95}")
    print(f"  AUDIO SANITY COMPARISON (VAL)")
    print(f"{'-' * 95}")
    print(f"  {'Variant':<55}" + "".join(f"{h:>10}" for h in header))
    print(f"{'-' * 95}")
    for variant, m in summary.items():
        print(f"  {variant:<55}" + "".join(f"{m[h]:>10}" for h in header))
    print(f"{'=' * 95}")


if __name__ == "__main__":
    main()
