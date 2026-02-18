import os, json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

from model_clients import (
    call_gpt,
    call_o3,
    call_o3_mini,
    call_referee,
    call_referee_audio,          # 🔥 NEW audio referee
    build_round1_prompt,
    build_round2_prompt,
    build_referee_prompt,  # 🔥 NEW referee prompt
    transcribe_audio
)

from utils import safe_parse_json, extract_label


LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "debate_gpt_audio_referee.jsonl"


# ----------------------------------------------------------
# TEXT-ONLY EXAMINERS
# ----------------------------------------------------------
GPT_MODELS = [
    ("o3", call_o3),
    ("o3_mini", call_o3_mini),
    ("gpt4o_text", call_gpt),   # gpt-4o in TEXT mode
]


# Allowed CEFR sequence for clamping referee output
CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]
def clamp_cefr(ref_label, baseline_label):
    """Allow referee to modify ±1 CEFR level only."""
    if ref_label not in CEFR_ORDER or baseline_label not in CEFR_ORDER:
        return baseline_label

    idx_base = CEFR_ORDER.index(baseline_label)
    idx_ref = CEFR_ORDER.index(ref_label)

    if abs(idx_ref - idx_base) <= 1:
        return ref_label
    return baseline_label

CEFR_TO_NUM = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5, 
    "C2": 6
}


# ----------------------------------------------------------
# Debate Engine (R1 + R2 + Referee)
# ----------------------------------------------------------
def multi_gpt_debate(transcript):

    log_data = {"round1": [], "round2": []}

    # ---------------------------
    # ROUND 1
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
    # ROUND 2
    # ---------------------------
    r2_outputs = []
    for name, fn in GPT_MODELS:

        self_entry = next(o for o in r1_outputs if o["model"] == name)
        others = [o for o in r1_outputs if o["model"] != name]

        prompt = build_round2_prompt(
            transcript,
            self_entry["label"],
            self_entry["parsed"].get("reason", ""),
            [{"model": o["model"], "label": o["label"], "reason": o["parsed"].get("reason","")} for o in others]
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
    # Majority vote on Round 2
    # ---------------------------
    votes = {}
    for o in r2_outputs:
        l = o["updated_label"]
        votes[l] = votes.get(l, 0) + 1

    majority_label = sorted(votes.items(), key=lambda x: -x[1])[0][0]
    log_data["majority_vote"] = votes
    log_data["majority_label"] = majority_label


    # ---------------------------
    # AUDIO REFEREE (FINAL DECISION)
    # ---------------------------
    ref_prompt = build_referee_prompt(
        transcript=transcript,
        round2_outputs=r2_outputs,
    )

    #ref_raw = call_referee_audio(prompt=ref_prompt, audio_path=audio_path)
    ref_raw = call_referee(ref_prompt)
    ref_parsed = safe_parse_json(ref_raw)
    ref_label = extract_label(ref_parsed)

    # apply ±1 constraint
    final_label = clamp_cefr(ref_label, majority_label)

    log_data["referee"] = {
        "raw": ref_raw,
        "parsed": ref_parsed,
        "raw_label": ref_label,
        "clamped_final_label": final_label
    }

    log_data["final_label"] = final_label
    return final_label, log_data




def compute_metrics(y_true, y_pred):
    """
    Computes ACC, ADJ, RMSE, PCC exactly as used in CEFR evaluation papers.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---- ACC ----
    acc = np.mean(y_true == y_pred)

    # ---- ADJ (Adjacent Accuracy: within ±1 band) ----
    adj = np.mean(np.abs(y_true - y_pred) <= 1)

    # ---- RMSE ----
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # ---- PCC (Pearson Correlation) ----
    pcc, _ = pearsonr(y_true, y_pred)

    return {
        "ACC": acc,
        "ADJ": adj,
        "RMSE": rmse,
        "PCC": pcc
    }




def main():

    df = pd.read_csv("../DebateDatasets/train_with_transcripts.csv")


    correct = 0
    total = 0

    # For PCC, RMSE, ACC, ADJ
    y_true_all = []
    y_pred_all = []

    RUNNING_METRICS_FILE = LOG_DIR / "running_metrics.json"
    SAVE_EVERY_N = 25   # save metrics every 25 samples safely

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        for idx, row in tqdm(df.iterrows(), total=len(df)):

            audio_path = row["audio_path"]
            truth = row["cefr_label"]

            # ---- Transcribe once only (GPU-safe) ----
            #transcript = transcribe_audio(audio_path)
            transcript = str(row["transcript"]).strip()


            if not transcript.strip():
                print(f"⚠️ Skipping empty transcript at index {idx}")
                continue

            # ---- Run full debate + audio referee ----
            pred, log_data = multi_gpt_debate(transcript)

            # ---- Logging (Crash-safe) ----
            log_data["sample_index"] = idx
            log_data["transcript"] = transcript
            log_data["ground_truth"] = truth

            f.write(json.dumps(log_data, ensure_ascii=False, indent=2) + "\n\n")

            # ---- Accuracy ----
            if pred == truth:
                correct += 1

            # ---- Ordinal Metrics Storage ----
            if truth in CEFR_TO_NUM and pred in CEFR_TO_NUM:
                y_true_all.append(CEFR_TO_NUM[truth])
                y_pred_all.append(CEFR_TO_NUM[pred])

            total += 1
            print(f"Truth={truth} | Pred={pred}")

            # ✅ ✅ ✅ ---- CRASH-SAFE RUNNING METRICS SAVE ---- ✅ ✅ ✅
            if total % SAVE_EVERY_N == 0:
                running_metrics = compute_metrics(y_true_all, y_pred_all)
                running_metrics["samples_completed"] = total

                with open(RUNNING_METRICS_FILE, "w") as rf:
                    json.dump(running_metrics, rf, indent=2)

                print(f" Saved running metrics at sample {total}")

    # ---- FINAL METRICS ----
    metrics = compute_metrics(y_true_all, y_pred_all)

    print("\n========== FINAL EVALUATION ==========")
    print("ACC  :", metrics["ACC"])
    print("ADJ  :", metrics["ADJ"])
    print("RMSE :", metrics["RMSE"])
    print("PCC  :", metrics["PCC"])

    # ✅ Final paper metrics
    with open(LOG_DIR / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
