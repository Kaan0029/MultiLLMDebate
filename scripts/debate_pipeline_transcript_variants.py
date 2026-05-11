# scripts/debate_pipeline_transcript_variants.py
"""
Transcript-only Multi-LLM Debate Pipeline with configurable few-shot examples.
Tests 1, 2, 4, and 8 in-context examples per class.
Uses same debate structure as debate_pipeline_gpt_only.py.
Supports optional referee mode for each variant.
Results saved to logs/transcript_variants/
"""
import os, json, argparse, time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_clients import (
    call_gpt, call_o3, call_o3_mini,
    build_round2_prompt, call_referee, build_referee_prompt,
)
from prompts_few_shot import build_round1_prompt_few_shot
from utils import safe_parse_json, extract_label
from metrics import (
    compute_metrics, print_metrics_table,
    error_analysis_report, analyse_round1, analyse_round2, analyse_vote, analyse_referee,
    CEFR_TO_NUM, VALID_CEFR, CEFR_ORDER,
)

LOG_DIR = Path("../logs/transcript_variants")
LOG_DIR.mkdir(parents=True, exist_ok=True)

GPT_MODELS = [
    ("gpt4o",   call_gpt),
    ("o3",      call_o3),
    ("o3_mini", call_o3_mini),
]
MODEL_ORDER = [name for name, _ in GPT_MODELS]

VAL_CSV = "../BaselineDatasets/dev_with_transcripts.csv"


def clamp_cefr(ref_label, baseline_label):
    if ref_label not in VALID_CEFR or baseline_label not in VALID_CEFR:
        return baseline_label
    idx_base = CEFR_ORDER.index(baseline_label)
    idx_ref = CEFR_ORDER.index(ref_label)
    if abs(idx_ref - idx_base) <= 1:
        return ref_label
    return baseline_label


def multi_gpt_debate(transcript, sample_index, n_examples, use_referee=False):
    log_data = {"round1": [], "round2": []}
    print(f"\n  ── Sample {sample_index} ──────────────────────────────")

    # Round 1 — parallel
    def call_r1(name, fn):
        prompt = build_round1_prompt_few_shot(transcript, n_examples)
        raw    = fn(prompt)
        parsed = safe_parse_json(raw)
        label  = extract_label(parsed)
        if label not in VALID_CEFR:
            print(f"    [R1] ⚠ {name} invalid label: '{label}'")
        return {"model": name, "raw": raw, "parsed": parsed, "label": label}

    r1_outputs = []
    with ThreadPoolExecutor(max_workers=len(GPT_MODELS)) as executor:
        futures = {executor.submit(call_r1, name, fn): name for name, fn in GPT_MODELS}
        for future in as_completed(futures):
            r1_outputs.append(future.result())
    r1_outputs.sort(key=lambda x: MODEL_ORDER.index(x["model"]))
    log_data["round1"] = r1_outputs
    r1_summary = analyse_round1(r1_outputs)

    time.sleep(2)

    # Round 2 — parallel
    def call_r2(name, fn):
        self_entry = next(o for o in r1_outputs if o["model"] == name)
        others     = [o for o in r1_outputs if o["model"] != name]
        prompt = build_round2_prompt(
            transcript,
            self_entry["label"],
            self_entry["parsed"].get("reason", ""),
            [{"model": o["model"], "label": o["label"],
              "reason": o["parsed"].get("reason", "")} for o in others],
        )
        raw           = fn(prompt)
        parsed        = safe_parse_json(raw)
        updated_label = extract_label(parsed)
        if updated_label not in VALID_CEFR:
            print(f"    [R2] ⚠ {name} invalid label: '{updated_label}'")
        return {"model": name, "raw": raw, "parsed": parsed, "updated_label": updated_label}

    r2_outputs = []
    with ThreadPoolExecutor(max_workers=len(GPT_MODELS)) as executor:
        futures = {executor.submit(call_r2, name, fn): name for name, fn in GPT_MODELS}
        for future in as_completed(futures):
            r2_outputs.append(future.result())
    r2_outputs.sort(key=lambda x: MODEL_ORDER.index(x["model"]))
    log_data["round2"] = r2_outputs
    r2_summary = analyse_round2(r1_outputs, r2_outputs)

    # Majority vote
    votes = {}
    for o in r2_outputs:
        lbl = o["updated_label"]
        votes[lbl] = votes.get(lbl, 0) + 1
    majority_label = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
    log_data["majority_vote"] = votes
    log_data["majority_label"] = majority_label
    log_data["final_label"] = majority_label
    analyse_vote(votes, majority_label)

    if use_referee:
        ref_prompt = build_referee_prompt(transcript=transcript, round2_outputs=r2_outputs)
        ref_raw = call_referee(ref_prompt)
        ref_parsed = safe_parse_json(ref_raw)
        ref_label = extract_label(ref_parsed)
        final_label = clamp_cefr(ref_label, majority_label)
        log_data["referee"] = {
            "raw": ref_raw,
            "parsed": ref_parsed,
            "raw_label": ref_label,
            "clamped_final_label": final_label,
        }
        log_data["final_label"] = final_label
        analyse_referee(majority_label, ref_label, final_label)
    else:
        final_label = majority_label

    step_summary = {
        "round1_labels":     r1_summary["labels"],
        "round2_labels":     r2_summary["r2_labels"],
        "vote_distribution": votes,
        "r1_unique_count":   r1_summary["unique_count"],
        "r2_changes":        r2_summary["changes"],
        "majority_pred":     majority_label,
        "ref_raw_label":     log_data.get("referee", {}).get("raw_label"),
    }
    return final_label, majority_label, log_data, step_summary


def run_variant(n_examples, use_referee=False):
    base_tag = f"transcript_{n_examples}ex"
    tag = f"{base_tag}_referee" if use_referee else base_tag
    print(f"\n{'='*60}")
    print(f"  TRANSCRIPT VARIANT — {n_examples} examples per class ({n_examples*5} total)")
    print(f"  Referee: {'enabled' if use_referee else 'disabled'}")
    print(f"{'='*60}")

    df = pd.read_csv(VAL_CSV)
    print(f"  Total samples: {len(df)}")

    log_file         = LOG_DIR / f"{tag}_logs.jsonl"
    result_file      = LOG_DIR / f"{tag}_results.json"
    running_met_file = LOG_DIR / f"{tag}_running_metrics.json"
    SAVE_EVERY_N     = 25

    y_true_all = []
    y_pred_final_all = []
    y_pred_majority_all = []
    sample_details = []
    skipped = 0

    with open(log_file, "w", encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=tag):
            truth      = str(row["cefr_label"]).strip()
            transcript = str(row.get("transcript", "")).strip()

            if not transcript:
                skipped += 1
                continue
            if truth not in VALID_CEFR:
                skipped += 1
                continue

            pred, majority_pred, log_data, step_summary = multi_gpt_debate(
                transcript, idx, n_examples, use_referee=use_referee
            )

            match_str = "✓" if pred == truth else "✗"
            if use_referee:
                print(f"  [{match_str}] Truth={truth}  Majority={majority_pred}  Final(ref)={pred}")
            else:
                print(f"  [{match_str}] Truth={truth}  Pred={pred}")

            log_data["sample_index"] = idx
            log_data["ground_truth"] = truth
            f.write(json.dumps(log_data, ensure_ascii=False, indent=2) + "\n\n")

            if pred in CEFR_TO_NUM:
                y_true_all.append(CEFR_TO_NUM[truth])
                y_pred_final_all.append(CEFR_TO_NUM[pred])
                y_pred_majority_all.append(CEFR_TO_NUM[majority_pred])
            else:
                print(f"  [WARN] Index {idx}: pred '{pred}' not in CEFR_TO_NUM")

            sample_details.append({
                "index":              idx,
                "truth":              truth,
                "pred":               pred,
                "majority_pred":      majority_pred,
                "ref_raw_label":      step_summary["ref_raw_label"],
                "transcript_snippet": transcript[:250],
                "vote_distribution":  step_summary["vote_distribution"],
                **{k: step_summary[k] for k in
                   ["round1_labels", "round2_labels", "r1_unique_count", "r2_changes"]},
            })

            n_completed = len(y_pred_final_all)
            if n_completed > 0 and n_completed % SAVE_EVERY_N == 0:
                running = compute_metrics(y_true_all, y_pred_final_all)
                running["samples_completed"] = n_completed
                with open(running_met_file, "w") as rf:
                    json.dump(running, rf, indent=2)

    print(f"\n  Skipped {skipped} samples.")
    metrics_final = compute_metrics(y_true_all, y_pred_final_all)
    print_metrics_table(metrics_final, title=f"{tag} FINAL  (n={len(y_true_all)})")
    if use_referee:
        metrics_majority = compute_metrics(y_true_all, y_pred_majority_all)
        print_metrics_table(metrics_majority, title=f"{tag} MAJORITY (pre-referee)")
    else:
        metrics_majority = metrics_final

    result = {
        "variant":   tag,
        "n_examples_per_class": n_examples,
        "n_samples": len(y_true_all),
        "skipped":   skipped,
        "metrics":   metrics_final,
        "metrics_majority": metrics_majority if use_referee else None,
    }
    with open(result_file, "w") as rf:
        json.dump(result, rf, indent=2)
    print(f"  Results saved → {result_file}")

    mode_name = f"Transcript {n_examples}-shot + referee" if use_referee else f"Transcript {n_examples}-shot"
    error_analysis_report(sample_details, pipeline_name=mode_name)
    return metrics_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_examples",
        choices=["1", "2", "4", "8", "all"],
        default="all",
        help="Number of in-context examples per class. 'all' runs 1,2,4,8 sequentially."
    )
    parser.add_argument(
        "--use_referee",
        action="store_true",
        help="Enable referee for each transcript variant."
    )
    args = parser.parse_args()

    variants = [1, 2, 4, 8] if args.n_examples == "all" else [int(args.n_examples)]

    all_results = {}
    for n in variants:
        metrics = run_variant(n, use_referee=args.use_referee)
        variant_key = f"transcript_{n}ex_referee" if args.use_referee else f"transcript_{n}ex"
        all_results[variant_key] = metrics

    summary_filename = "transcript_variants_referee_summary.json" if args.use_referee else "transcript_variants_summary.json"
    summary_file = LOG_DIR / summary_filename
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved → {summary_file}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  TRANSCRIPT VARIANTS COMPARISON")
    print(f"{'─'*70}")
    header = ["ACC", "ACC_ADJ", "ACC_MC", "ACC_MC_ADJ", "RMSE", "RMSE_MC", "PCC"]
    print(f"  {'Variant':<22}" + "".join(f"{m:>10}" for m in header))
    print(f"{'─'*70}")
    for variant, m in all_results.items():
        print(f"  {variant:<22}" + "".join(f"{m[k]:>10}" for k in header))
    print(f"{'='*70}")


if __name__ == "__main__":
    main()