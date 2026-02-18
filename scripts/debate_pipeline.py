# debate_pipeline.py
import os, json
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from model_clients import (
    call_gpt, call_claude, call_gemini, call_grok, call_referee,
    build_round1_prompt, build_round2_prompt, build_referee_prompt
)
from utils import safe_parse_json, extract_label

LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "debate_full_logs_majority.jsonl"

MODELS = [
    ("gpt4o", call_gpt),
    ("claude", call_claude),
    ("gemini", call_gemini),
    ("grok", call_grok),
]

# ----------------------------------------------------------
# Debate: Round 1 + Round 2
# ----------------------------------------------------------
def multi_llm_debate(transcript):

    log_data = {"round1": [], "round2": []}

    # ---------------------------
    # ROUND 1: independent judgments
    # ---------------------------
    r1_outputs = []

    for name, fn in MODELS:
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

    for name, fn in MODELS:

        self_entry = next(o for o in r1_outputs if o["model"] == name)
        others = [o for o in r1_outputs if o["model"] != name]

        prompt = build_round2_prompt(
            transcript,
            self_entry["label"],
            self_entry["parsed"].get("reason", ""),
            [{"model": o["model"], "label": o["label"], "reason": o["parsed"].get("reason", "")} for o in others]
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
    # Majority vote on updated labels --> (can consider using a refree model, single model that makes a decision in the end)
    # ---------------------------
    votes = {}
    for o in r2_outputs:
        l = o["updated_label"]
        votes[l] = votes.get(l, 0) + 1

    final = sorted(votes.items(), key=lambda x: -x[1])[0][0]
    log_data["majority_vote"] = votes
    log_data["final_label"] = final

    # return final, log_data

    # ---------------------------
    # Referee Model Decision
    # ---------------------------
    # ref_prompt = build_referee_prompt(transcript, r2_outputs)
    # ref_raw = call_referee(ref_prompt)
    # ref_parsed = safe_parse_json(ref_raw)
    # final = extract_label(ref_parsed)

    # log_data["referee"] = {
    #     "model": "gpt4o",
    #     "raw": ref_raw,
    #     "parsed": ref_parsed,
    #     "final_label": final
    # }

    return final, log_data




# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main():
    df = pd.read_csv("../data/icnale_subset.csv")

    correct = 0
    total = 0

    with open(LOG_FILE, "w", encoding="utf-8") as f:

        for idx, row in tqdm(df.iterrows(), total=len(df)):

            audio_path = row["audio_path"]
            truth = row["cefr_label"]

            from model_clients import transcribe_audio
            transcript = transcribe_audio(audio_path)


            pred, log_data = multi_llm_debate(transcript)

            log_data["sample_index"] = idx
            log_data["audio_path"] = audio_path
            log_data["ground_truth"] = truth

            f.write(json.dumps(log_data, ensure_ascii=False, indent=2) + "\n\n")

            if pred == truth:
                correct += 1
            total += 1

            print(f"Truth={truth} | Pred={pred}")

    print("\nFinal accuracy:", correct / total)


if __name__ == "__main__":
    main()


'''
Different approach: use a refree model as the final decision making 
make different experiments with this audio + gpt 4o and prepare performance report 
300 data sample --> ICNALE 
'''
