import pandas as pd
from tqdm import tqdm
from pathlib import Path

from model_clients import transcribe_audio

INPUT_CSV = "../DebateDatasets/test.csv"
OUTPUT_CSV = "../DebateDatasets/test_with_transcripts.csv"

df = pd.read_csv(INPUT_CSV)

# If resuming, skip already-done rows
if "transcript" not in df.columns:
    df["transcript"] = ""

print(f"Loaded {len(df)} rows")

for idx in tqdm(range(len(df))):
    if df.loc[idx, "transcript"].strip():
        continue  # already done

    audio_path = df.loc[idx, "audio_path"]

    try:
        text = transcribe_audio(audio_path)
        df.loc[idx, "transcript"] = text
    except Exception as e:
        print(f"❌ Failed on {audio_path}: {e}")
        df.loc[idx, "transcript"] = ""

    # 💾 SAVE EVERY 10 FILES (CRASH SAFE)
    if idx % 10 == 0:
        df.to_csv(OUTPUT_CSV, index=False)

# Final save
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved transcripts to {OUTPUT_CSV}")
