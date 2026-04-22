# scripts/debate_pipeline_gpt_audio_referee.py
"""
GPT + Referee Multi-LLM Debate Pipeline.
Models: o3 + o3-mini + GPT-4o (text) for debate; GPT-4o as referee.
Uses ThreadPoolExecutor to call all 3 debate models in parallel within each round.
Referee call is sequential (depends on R2 results).
Referee decision is clamped to ±1 CEFR level from the majority vote.
Supports 3 data splits: val_only | train_only | train_val
Outputs all 7 evaluation metrics + full error analysis.
"""
import os, json, argparse, time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_clients import (
    call_gpt,
    call_o3,
    call_o3_mini,
    call_referee,
    build_round1_prompt,
    build_round2_prompt,
    build_referee_prompt,
)
from utils import safe_parse_json, extract_label
from metrics import (
    compute_metrics,
    print_metrics_table,
    error_analysis_report,
    analyse_round1,
    analyse_round2,
    analyse_vote,
    analyse_referee,
    CEFR_TO_NUM,
    CEFR_ORDER,
    VALID_CEFR,
)

# ─────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────

LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)

GPT_MODELS = [
    ("o3",         call_o3),
    ("o3_mini",    call_o3_mini),
    ("gpt4o_text", call_gpt),
]

MODEL_ORDER = [name for name, _ in GPT_MODELS]

DATA_SPLITS = {
    "val_only":   [
        "../BaselineDatasets/dev_with_transcripts.csv",
    ],
    "train_only": [
        "../BaselineDatasets/train_with_transcripts.csv",
    ],
    "train_val":  [
        "../BaselineDatasets/train_with_transcripts.csv",
        "../BaselineDatasets/dev_with_transcripts.csv",
    ],
}


# ─────────────────────────────────────────────────────────
# Referee clamp helper
# ─────────────────────────────────────────────────────────

def clamp_cefr(ref_label, baseline_label):
    """
    Restricts referee to change the majority label by at most ±1 CEFR level.
    If ref_label is more than 1 step away, returns baseline_label unchanged.
    If either label is invalid, returns baseline_label as the safe fallback.
    """
    if ref_label not in VALID_CEFR or baseline_label not in VALID_CEFR:
        print(f"    [REF] ⚠ Clamp received invalid label(s): "
              f"ref='{ref_label}', baseline='{baseline_label}' — returning baseline.")
        return baseline_label

    idx_base = CEFR_ORDER.index(baseline_label)
    idx_ref  = CEFR_ORDER.index(ref_label)

    if abs(idx_ref - idx_base) <= 1:
        return ref_label          # referee's suggestion accepted
    else:
        return baseline_label     # clamp triggered — too far from majority


# ─────────────────────────────────────────────────────────
# Debate engine (Round 1 + Round 2 + majority + referee)
# R1 and R2 model calls are parallelised.
# Referee call is sequential (depends on R2 outputs).
# ─────────────────────────────────────────────────────────

def multi_gpt_referee_debate(transcript, sample_index):
    """
    Runs a two-round debate among GPT_MODELS, then passes the result
    to a referee model for a final constrained decision.
    Within each round, all 3 models are called in parallel.

    Returns
    -------
    final_label    : str    — referee-adjusted CEFR label (clamped to ±1)
    majority_label : str    — raw majority vote label (pre-referee)
    log_data       : dict   — full structured log for this sample
    step_summary   : dict   — compact summary for post-hoc error analysis
    """
    log_data = {"round1": [], "round2": []}
    print(f"\n  ── Sample {sample_index} ──────────────────────────────")

    # ── Round 1: All 3 models called IN PARALLEL ──────────────────────────
    def call_r1(name, fn):
        prompt = build_round1_prompt(transcript)
        raw    = fn(prompt)
        parsed = safe_parse_json(raw)
        label  = extract_label(parsed)
        if label not in VALID_CEFR:
            print(f"    [R1] ⚠ {name} returned invalid/unparseable label: '{label}' — "
                  f"raw snippet: {raw[:80]!r}")
        return {
            "model":  name,
            "raw":    raw,
            "parsed": parsed,
            "label":  label,
        }

    r1_outputs = []
    with ThreadPoolExecutor(max_workers=len(GPT_MODELS)) as executor:
        futures = {
            executor.submit(call_r1, name, fn): name
            for name, fn in GPT_MODELS
        }
        for future in as_completed(futures):
            r1_outputs.append(future.result())

    r1_outputs.sort(key=lambda x: MODEL_ORDER.index(x["model"]))
    log_data["round1"] = r1_outputs

    # Step-by-step R1 analysis
    r1_summary = analyse_round1(r1_outputs)

    time.sleep(2)

    # ── Round 2: All 3 models called IN PARALLEL ──────────────────────────
    # Each model sees all R1 outputs (including others') before responding.
    # All 3 R2 calls are independent of each other so can be parallelised.
    def call_r2(name, fn):
        self_entry = next(o for o in r1_outputs if o["model"] == name)
        others     = [o for o in r1_outputs if o["model"] != name]
        prompt = build_round2_prompt(
            transcript,
            self_entry["label"],
            self_entry["parsed"].get("reason", ""),
            [
                {
                    "model":  o["model"],
                    "label":  o["label"],
                    "reason": o["parsed"].get("reason", ""),
                }
                for o in others
            ],
        )
        raw           = fn(prompt)
        parsed        = safe_parse_json(raw)
        updated_label = extract_label(parsed)
        if updated_label not in VALID_CEFR:
            print(f"    [R2] ⚠ {name} returned invalid/unparseable label: '{updated_label}' — "
                  f"raw snippet: {raw[:80]!r}")
        return {
            "model":         name,
            "raw":           raw,
            "parsed":        parsed,
            "updated_label": updated_label,
        }

    r2_outputs = []
    with ThreadPoolExecutor(max_workers=len(GPT_MODELS)) as executor:
        futures = {
            executor.submit(call_r2, name, fn): name
            for name, fn in GPT_MODELS
        }
        for future in as_completed(futures):
            r2_outputs.append(future.result())

    r2_outputs.sort(key=lambda x: MODEL_ORDER.index(x["model"]))
    log_data["round2"] = r2_outputs

    # Step-by-step R2 analysis
    r2_summary = analyse_round2(r1_outputs, r2_outputs)

    # ── Majority vote on Round 2 labels ───────────────────────────────────
    votes = {}
    for o in r2_outputs:
        lbl = o["updated_label"]
        votes[lbl] = votes.get(lbl, 0) + 1

    majority_label = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
    log_data["majority_vote"]  = votes
    log_data["majority_label"] = majority_label

    # Step-by-step vote analysis
    analyse_vote(votes, majority_label)

    time.sleep(2)

    # ── Referee: final constrained decision (sequential — depends on R2) ──
    ref_prompt = build_referee_prompt(
        transcript=transcript,
        round2_outputs=r2_outputs,
    )

    ref_raw    = call_referee(ref_prompt)
    ref_parsed = safe_parse_json(ref_raw)
    ref_label  = extract_label(ref_parsed)

    if ref_label not in VALID_CEFR:
        print(f"    [REF] ⚠ Referee returned invalid/unparseable label: '{ref_label}' — "
              f"raw snippet: {ref_raw[:80]!r}  — will fall back to majority.")

    # Apply ±1 clamp constraint
    final_label = clamp_cefr(ref_label, majority_label)

    log_data["referee"] = {
        "raw":                 ref_raw,
        "parsed":              ref_parsed,
        "raw_label":           ref_label,
        "clamped_final_label": final_label,
    }
    log_data["final_label"] = final_label

    # Step-by-step referee analysis
    analyse_referee(majority_label, ref_label, final_label)

    # ── Build step summary for post-hoc error analysis ────────────────────
    step_summary = {
        "round1_labels":     r1_summary["labels"],
        "round2_labels":     r2_summary["r2_labels"],
        "vote_distribution": votes,
        "r1_unique_count":   r1_summary["unique_count"],
        "r2_changes":        r2_summary["changes"],
        "majority_pred":     majority_label,
        "ref_raw_label":     ref_label,
    }

    return final_label, majority_label, log_data, step_summary


# ─────────────────────────────────────────────────────────
# Run one data split end-to-end
# ─────────────────────────────────────────────────────────

def run_split(split_name, csv_paths, log_dir="../logs"):
    LOG_DIR = Path(log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  GPT + REFEREE DEBATE PIPELINE")
    print(f"  Split : {split_name}")
    print(f"  CSVs  : {csv_paths}")
    print(f"{'='*60}")

    # Load and concatenate split CSVs
    frames = []
    for path in csv_paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p.resolve()}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total samples loaded: {len(df)}")

    # Validate expected columns
    required_cols = {"cefr_label", "transcript"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}. "
                         f"Found columns: {list(df.columns)}")

    # Output file paths for this split
    log_file         = LOG_DIR / f"gpt_referee_{split_name}_logs.jsonl"
    result_file      = LOG_DIR / f"gpt_referee_{split_name}_results.json"
    running_met_file = LOG_DIR / f"gpt_referee_{split_name}_running_metrics.json"
    SAVE_EVERY_N     = 25

    # Track metrics for both the final (referee) and majority-only predictions
    # for ablation comparison (did the referee actually help?)
    y_true_all          = []
    y_pred_final_all    = []   # referee-adjusted final predictions
    y_pred_majority_all = []   # raw majority vote predictions (pre-referee)
    sample_details      = []
    skipped             = 0

    with open(log_file, "w", encoding="utf-8") as f:

        for idx, row in tqdm(df.iterrows(), total=len(df),
                             desc=f"gpt_referee/{split_name}"):

            truth      = str(row["cefr_label"]).strip().upper()
            if truth in ("B1_1", "B1_2"):
                truth = "B1"
            transcript = str(row.get("transcript", "")).strip()

            # ── Skip guard ────────────────────────────────────────────────
            if not transcript:
                print(f"  [SKIP] Index {idx}: empty transcript.")
                skipped += 1
                continue
            if truth not in VALID_CEFR:
                print(f"  [SKIP] Index {idx}: invalid ground-truth label '{truth}'.")
                skipped += 1
                continue

            # ── Run debate + referee ──────────────────────────────────────
            pred, majority_pred, log_data, step_summary = \
                multi_gpt_referee_debate(transcript, idx)

            # ── Per-sample result print ───────────────────────────────────
            match_final    = "✓" if pred          == truth else "✗"
            match_majority = "✓" if majority_pred == truth else "✗"
            print(f"  [{match_final}] Truth={truth}  "
                  f"Majority={majority_pred}[{match_majority}]  "
                  f"Final(referee)={pred}[{match_final}]")

            # ── Structured log write ──────────────────────────────────────
            log_data["sample_index"] = idx
            log_data["ground_truth"] = truth
            f.write(json.dumps(log_data, ensure_ascii=False, indent=2) + "\n\n")

            # ── Metrics accumulation ──────────────────────────────────────
            if pred in CEFR_TO_NUM and majority_pred in CEFR_TO_NUM:
                y_true_all.append(CEFR_TO_NUM[truth])
                y_pred_final_all.append(CEFR_TO_NUM[pred])
                y_pred_majority_all.append(CEFR_TO_NUM[majority_pred])
            else:
                print(f"  [WARN] Index {idx}: pred='{pred}' or majority='{majority_pred}' "
                      f"not in CEFR_TO_NUM — excluded from metric computation.")

            # ── Sample details for post-hoc analysis ─────────────────────
            sample_details.append({
                "index":              idx,
                "truth":              truth,
                "pred":               pred,
                "majority_pred":      majority_pred,
                "ref_raw_label":      step_summary["ref_raw_label"],
                "transcript_snippet": transcript[:250],
                "vote_distribution":  step_summary["vote_distribution"],
                **{k: step_summary[k] for k in
                   ["round1_labels", "round2_labels",
                    "r1_unique_count", "r2_changes"]},
            })

            # ── Crash-safe running metrics ────────────────────────────────
            n_completed = len(y_true_all)
            if n_completed > 0 and n_completed % SAVE_EVERY_N == 0:
                running = compute_metrics(y_true_all, y_pred_final_all)
                running["samples_completed"] = n_completed
                with open(running_met_file, "w") as rf:
                    json.dump(running, rf, indent=2)
                print(f"\n  [Checkpoint] Running metrics saved at {n_completed} samples.")

    # ── Final metrics ──────────────────────────────────────────────────────
    print(f"\n  Skipped {skipped} samples (empty or invalid labels).")

    metrics_final    = compute_metrics(y_true_all, y_pred_final_all)
    metrics_majority = compute_metrics(y_true_all, y_pred_majority_all)

    print_metrics_table(
        metrics_final,
        title=f"gpt_referee / {split_name}  (FINAL — after referee)  n={len(y_true_all)}"
    )
    print_metrics_table(
        metrics_majority,
        title=f"gpt_referee / {split_name}  (MAJORITY only — pre-referee ablation)"
    )

    # ── Save results ───────────────────────────────────────────────────────
    result = {
        "pipeline":         "gpt_referee",
        "split":            split_name,
        "n_samples":        len(y_true_all),
        "skipped":          skipped,
        "metrics_final":    metrics_final,
        "metrics_majority": metrics_majority,
    }
    with open(result_file, "w") as rf:
        json.dump(result, rf, indent=2)
    print(f"  Results saved → {result_file}")
    print(f"  Full logs saved → {log_file}")

    # ── Post-hoc error analysis ────────────────────────────────────────────
    error_analysis_report(
        sample_details,
        pipeline_name=f"GPT+Referee / {split_name}",
    )

    return metrics_final


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run GPT+Referee Multi-LLM Debate pipeline for CEFR classification."
    )
    parser.add_argument(
        "--split",
        choices=["val_only", "train_only", "train_val", "all"],
        default="val_only",
        help=(
            "Data split to evaluate on.\n"
            "  val_only   — ICNALE validation set only\n"
            "  train_only — ICNALE training set only\n"
            "  train_val  — training + validation combined\n"
            "  all        — run all three splits sequentially"
        ),
    )
    parser.add_argument("--logdir", default="../logs",
                        help="Output directory for logs and results")
    args = parser.parse_args()

    splits_to_run = (
        list(DATA_SPLITS.keys()) if args.split == "all" else [args.split]
    )

    all_results = {}
    for split_name in splits_to_run:
        metrics = run_split(split_name, DATA_SPLITS[split_name], args.logdir)
        all_results[split_name] = metrics

    summary_file = Path(args.logdir) / "gpt_referee_all_splits_summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined summary saved → {summary_file}")

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  CROSS-SPLIT COMPARISON  —  GPT+Referee")
        print(f"{'─'*60}")
        header_metrics = ["ACC", "ACC_ADJ", "ACC_MC", "ACC_MC_ADJ", "RMSE", "RMSE_MC", "PCC"]
        print(f"  {'Split':<16}" + "".join(f"{m:>12}" for m in header_metrics))
        print(f"{'─'*60}")
        for split_name, m in all_results.items():
            row = f"  {split_name:<16}" + "".join(f"{m[k]:>12}" for k in header_metrics)
            print(row)
        print(f"{'='*60}")


if __name__ == "__main__":
    main()