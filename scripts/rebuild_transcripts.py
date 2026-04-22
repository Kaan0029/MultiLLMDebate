# scripts/rebuild_transcripts.py
"""
Rebuilds *_with_transcripts.csv files from the correctly-labelled base CSVs.
Uses pre-computed Whisper transcriptions where available to avoid re-transcribing.
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from model_clients import transcribe_audio

SPLITS = [
    ("../BaselineDatasets/dev.csv",   "../BaselineDatasets/dev_with_transcripts.csv"),
    ("../BaselineDatasets/train.csv", "../BaselineDatasets/train_with_transcripts.csv"),
    ("../BaselineDatasets/test.csv",  "../BaselineDatasets/test_with_transcripts.csv"),
]

for input_csv, output_csv in SPLITS:
    print(f"\nProcessing {input_csv} → {output_csv}")
    df = pd.read_csv(input_csv)

    if "transcript" not in df.columns:
        df["transcript"] = ""

    missing = df["transcript"].isna() | (df["transcript"].str.strip() == "")
    print(f"  Total: {len(df)}  |  Need transcription: {missing.sum()}")

    for idx in tqdm(df[missing].index):
        audio_path = df.loc[idx, "audio_path"]
        try:
            df.loc[idx, "transcript"] = transcribe_audio(audio_path)
        except Exception as e:
            print(f"  ❌ Failed on {audio_path}: {e}")
            df.loc[idx, "transcript"] = ""

        if idx % 10 == 0:
            df.to_csv(output_csv, index=False)

    df.to_csv(output_csv, index=False)
    print(f"  ✓ Saved {output_csv}")