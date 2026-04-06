import os, json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np                          # ADDED
from scipy.stats import pearsonr            # ADDED

from model_clients import (
    call_gpt,
    call_o3,
    call_o3_mini,
    build_round1_prompt,
    build_round2_prompt,
    # transcribe_audio   <-- REMOVED, using pre-computed transcripts instead
)

from utils import safe_parse_json, extract_label

LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "debate_gpt_only_logs.jsonl"

# ADDED: needed for ordinal metrics
CEFR_TO_NUM = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

# ----------------------------------------------------------
# GPT-ONLY MODELS
# ----------------------------------------------------------

GPT_MODELS = [
    ("gpt4o", call_gpt),
    ("o3", call_o3),
    ("o3_mini", call_o3_mini),
]

# ADDED: proper metrics function matching baseline table
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc  = np.mean(y_true == y_pred)
    adj  = np.mean(np.abs(y_true - y_pred) <= 1)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pcc, _ = pearsonr(y_true, y_pred)
    return {"ACC": acc, "ADJ": adj, "RMSE": rmse, "PCC": pcc}

# ----------------------------------------------------------
# Debate: Round 1 + Round 2
# ----------------------------------------------------------
def multi_gpt_debate(transcript):

    log_data = {"round1": [], "round2": []}

    # ---------------------------
    # ROUND 1: independent judgments
    # ---------------------------
    r1_outputs = []

    for name, fn in GPT_MODELS:

        prompt = build_round1_prompt(transcript)
        raw = fn(prompt)

        parsed = safe_parse_json(raw)
        label = extract_label(parsed)

        r1_outputs.append({
            "model": name,
            "raw": raw,
            "parsed": parsed,
            "label": label
        })

    log_data["round1"] = r1_outputs

    # ---------------------------
    # ROUND 2: debate + reconsideration
    # ---------------------------
    r2_outputs = []

    for name, fn in GPT_MODELS:

        self_entry = next(o for o in r1_outputs if o["model"] == name)
        others = [o for o in r1_outputs if o["model"] != name]

        prompt = build_round2_prompt(
            transcript,
            self_entry["label"],
            self_entry["parsed"].get("reason", ""),
            [
                {
                    "model": o["model"],
                    "label": o["label"],
                    "reason": o["parsed"].get("reason", "")
                }
                for o in others
            ]
        )

        raw = fn(prompt)
        parsed = safe_parse_json(raw)
        updated_label = extract_label(parsed)

        r2_outputs.append({
            "model": name,
            "raw": raw,
            "parsed": parsed,
            "updated_label": updated_label
        })

    log_data["round2"] = r2_outputs

    # ---------------------------
    # Majority vote
    # ---------------------------
    votes = {}

    for o in r2_outputs:
        l = o["updated_label"]
        votes[l] = votes.get(l, 0) + 1

    final_prediction = sorted(votes.items(), key=lambda x: -x[1])[0][0]

    log_data["majority_vote"] = votes
    log_data["final_label"] = final_prediction

    return final_prediction, log_data


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():

    df = pd.read_csv("../BaselineDatasets/dev_with_transcripts.csv")  # CHANGED from ../train.csv

    correct = 0
    total = 0
    y_true_all = []   # ADDED
    y_pred_all = []   # ADDED

    with open(LOG_FILE, "w", encoding="utf-8") as f:

        for idx, row in tqdm(df.iterrows(), total=len(df)):

            truth = row["cefr_label"]

            transcript = str(row["transcript"]).strip()   # CHANGED: use pre-computed instead of transcribe_audio
            if not transcript:                             # ADDED
                print(f"Skipping empty transcript at index {idx}")
                continue

            pred, log_data = multi_gpt_debate(transcript)

            log_data["sample_index"] = idx
            log_data["ground_truth"] = truth

            f.write(json.dumps(log_data, ensure_ascii=False, indent=2) + "\n\n")

            if pred == truth:
                correct += 1

            # ADDED: track ordinal values for metrics
            if truth in CEFR_TO_NUM and pred in CEFR_TO_NUM:
                y_true_all.append(CEFR_TO_NUM[truth])
                y_pred_all.append(CEFR_TO_NUM[pred])

            total += 1
            print(f"Truth = {truth} | Prediction = {pred}")

    # CHANGED: replaced simple accuracy print with full metrics
    metrics = compute_metrics(y_true_all, y_pred_all)
    print("\n========== FINAL EVALUATION ==========")
    print(f"ACC  : {metrics['ACC']:.4f}")
    print(f"ADJ  : {metrics['ADJ']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"PCC  : {metrics['PCC']:.4f}")

    with open(LOG_DIR / "gpt_only_val_metrics.json", "w") as f:   # ADDED: save to json
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()