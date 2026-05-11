# scripts/debate_pipeline_audio_only.py
"""
Audio-only Multi-LLM Debate Pipeline.
Uses gpt-4o-audio-preview x3 — receives raw audio, no transcript.
Includes 1 in-context audio example per class from the training set.
Supports optional GPT audio referee (--use_referee).
Results saved to logs/audio_only/
"""
import os, json, argparse, time
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from model_clients import call_gpt_audio, call_referee_audio
from prompts_audio import IN_CONTEXT_AUDIO_PATHS, AUDIO_ROUND1_PROMPT, AUDIO_ROUND2_TEMPLATE
from utils import safe_parse_json, extract_label
from metrics import (
    compute_metrics, print_metrics_table,
    error_analysis_report, analyse_round1, analyse_round2, analyse_vote, analyse_referee,
    CEFR_TO_NUM, VALID_CEFR, CEFR_ORDER,
)

LOG_DIR = Path("../logs/audio_only")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 3 independent instances of gpt-4o-audio-preview
AUDIO_MODELS = [
    ("gpt4o_audio_1", call_gpt_audio),
    ("gpt4o_audio_2", call_gpt_audio),
    ("gpt4o_audio_3", call_gpt_audio),
]
MODEL_ORDER = [name for name, _ in AUDIO_MODELS]

VAL_CSV = "../BaselineDatasets/dev_with_transcripts.csv"


def build_r2_prompt(self_label, self_reason, others):
    others_text = ""
    for o in others:
        others_text += f"- {o['model']}: label={o['label']}, reason={o['reason']}\n"
    return AUDIO_ROUND2_TEMPLATE.format(
        self_label=self_label,
        self_reason=self_reason,
        other_judgments=others_text,
    )


def build_audio_referee_prompt(round2_outputs):
    debate_text = ""
    for o in round2_outputs:
        debate_text += (
            f"- {o['model']} -> label: {o['updated_label']}\n"
            f"  reason: {o['parsed'].get('reason', '')}\n"
        )

    return f"""You are a senior examiner moderating spoken English proficiency labels.

The audio has already been listened to by all examiners.
You are given their round-2 labels and reasons.

Valid labels (only): A2, B1_1, B1_2, B2, native
Ordered levels: A2 < B1_1 < B1_2 < B2 < native

Decision rule:
- You may keep the majority level or adjust by at most one level.
- Adjust only if the panel reasoning is clearly inconsistent.

Panel judgments:
{debate_text}

Return ONLY JSON:
{{
  "final_label": "<A2/B1_1/B1_2/B2/native>",
  "reason": "<brief justification>"
}}"""


def clamp_cefr(ref_label, baseline_label):
    if ref_label not in VALID_CEFR or baseline_label not in VALID_CEFR:
        return baseline_label
    idx_base = CEFR_ORDER.index(baseline_label)
    idx_ref = CEFR_ORDER.index(ref_label)
    if abs(idx_ref - idx_base) <= 1:
        return ref_label
    return baseline_label


def multi_audio_debate(audio_path, sample_index, use_referee=False):
    log_data = {"round1": [], "round2": []}
    print(f"\n  ── Sample {sample_index} ──────────────────────────────")

    # Round 1 — all 3 models listen in parallel
    def call_r1(name, fn):
        raw    = fn(AUDIO_ROUND1_PROMPT, audio_path, IN_CONTEXT_AUDIO_PATHS)
        parsed = safe_parse_json(raw)
        label  = extract_label(parsed)
        if label not in VALID_CEFR:
            print(f"    [R1] ⚠ {name} invalid label: '{label}'")
        return {"model": name, "raw": raw, "parsed": parsed, "label": label}

    r1_outputs = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(call_r1, name, fn): name for name, fn in AUDIO_MODELS}
        for future in as_completed(futures):
            r1_outputs.append(future.result())
    r1_outputs.sort(key=lambda x: MODEL_ORDER.index(x["model"]))
    log_data["round1"] = r1_outputs
    r1_summary = analyse_round1(r1_outputs)

    time.sleep(2)

    # Round 2 — text-based reconsideration (audio already heard in R1)
    def call_r2(name, fn):
        self_entry = next(o for o in r1_outputs if o["model"] == name)
        others     = [o for o in r1_outputs if o["model"] != name]
        prompt = build_r2_prompt(
            self_entry["label"],
            self_entry["parsed"].get("reason", ""),
            [{"model": o["model"], "label": o["label"],
              "reason": o["parsed"].get("reason", "")} for o in others],
        )
        # R2 is text-only — no need to re-send audio
        raw    = fn(prompt, audio_path)   # audio_path still passed but prompt explains it was already heard
        parsed = safe_parse_json(raw)
        updated_label = extract_label(parsed)
        if updated_label not in VALID_CEFR:
            print(f"    [R2] ⚠ {name} invalid label: '{updated_label}'")
        return {"model": name, "raw": raw, "parsed": parsed, "updated_label": updated_label}

    r2_outputs = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(call_r2, name, fn): name for name, fn in AUDIO_MODELS}
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
        ref_prompt = build_audio_referee_prompt(r2_outputs)
        ref_raw = call_referee_audio(ref_prompt, audio_path)
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


def run(use_referee=False):
    mode_tag = "audio_only_referee" if use_referee else "audio_only"
    print(f"\n{'='*60}")
    print(f"  AUDIO-ONLY DEBATE PIPELINE")
    print(f"  Models: gpt-4o-audio-preview x3")
    print(f"  Referee: {'gpt-4o-audio-preview (enabled)' if use_referee else 'disabled'}")
    print(f"  In-context: 1 audio example per class")
    print(f"{'='*60}")

    df = pd.read_csv(VAL_CSV)
    print(f"  Total samples: {len(df)}")

    log_file = LOG_DIR / f"{mode_tag}_val_logs.jsonl"
    result_file = LOG_DIR / f"{mode_tag}_val_results.json"
    running_met_file = LOG_DIR / f"{mode_tag}_val_running_metrics.json"
    SAVE_EVERY_N     = 10   # smaller interval since audio calls are slower

    y_true_all = []
    y_pred_final_all = []
    y_pred_majority_all = []
    sample_details = []
    skipped = 0

    with open(log_file, "w", encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="audio_only"):
            truth      = str(row["cefr_label"]).strip()
            audio_path = str(row.get("audio_path", "")).strip()

            if not audio_path or not Path(audio_path).exists():
                print(f"  [SKIP] Index {idx}: missing audio.")
                skipped += 1
                continue
            if truth not in VALID_CEFR:
                print(f"  [SKIP] Index {idx}: invalid label '{truth}'.")
                skipped += 1
                continue

            pred, majority_pred, log_data, step_summary = multi_audio_debate(
                audio_path, idx, use_referee=use_referee
            )

            match_str = "✓" if pred == truth else "✗"
            if use_referee:
                print(f"  [{match_str}] Truth={truth}  Majority={majority_pred}  Final(ref)={pred}")
            else:
                print(f"  [{match_str}] Truth={truth}  Pred={pred}")

            log_data["sample_index"] = idx
            log_data["ground_truth"] = truth
            log_data["audio_path"]   = audio_path
            f.write(json.dumps(log_data, ensure_ascii=False, indent=2) + "\n\n")

            if pred in CEFR_TO_NUM:
                y_true_all.append(CEFR_TO_NUM[truth])
                y_pred_final_all.append(CEFR_TO_NUM[pred])
                y_pred_majority_all.append(CEFR_TO_NUM[majority_pred])
            else:
                print(f"  [WARN] Index {idx}: pred '{pred}' not in CEFR_TO_NUM")

            sample_details.append({
                "index":             idx,
                "truth":             truth,
                "pred":              pred,
                "majority_pred":     majority_pred,
                "ref_raw_label":     step_summary["ref_raw_label"],
                "audio_path":        audio_path,
                "vote_distribution": step_summary["vote_distribution"],
                **{k: step_summary[k] for k in
                   ["round1_labels", "round2_labels", "r1_unique_count", "r2_changes"]},
            })

            n_completed = len(y_pred_final_all)
            if n_completed > 0 and n_completed % SAVE_EVERY_N == 0:
                running = compute_metrics(y_true_all, y_pred_final_all)
                running["samples_completed"] = n_completed
                with open(running_met_file, "w") as rf:
                    json.dump(running, rf, indent=2)
                print(f"\n  [Checkpoint] Running metrics saved at {n_completed} samples.")

    print(f"\n  Skipped {skipped} samples.")
    metrics_final = compute_metrics(y_true_all, y_pred_final_all)
    print_metrics_table(metrics_final, title=f"{mode_tag} / val FINAL  (n={len(y_true_all)})")
    if use_referee:
        metrics_majority = compute_metrics(y_true_all, y_pred_majority_all)
        print_metrics_table(metrics_majority, title=f"{mode_tag} / val MAJORITY (pre-referee)")
    else:
        metrics_majority = metrics_final

    result = {
        "pipeline": mode_tag,
        "split":     "val_only",
        "n_samples": len(y_true_all),
        "skipped":   skipped,
        "metrics":   metrics_final,
        "metrics_majority": metrics_majority if use_referee else None,
    }
    with open(result_file, "w") as rf:
        json.dump(result, rf, indent=2)
    print(f"  Results saved → {result_file}")

    error_analysis_report(sample_details, pipeline_name="Audio-only")
    return metrics_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_referee",
        action="store_true",
        help="Enable GPT audio referee after majority vote."
    )
    args = parser.parse_args()
    run(use_referee=args.use_referee)


if __name__ == "__main__":
    main()