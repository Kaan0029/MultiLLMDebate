import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

TEXT_ROOT  = Path("/home/ke2461/MultiLLMDebate/ICNALE/SM_1_Classified_Unmerged")
AUDIO_ROOT = Path("/home/ke2461/MultiLLMDebate/ICNALE/ICNALE_SM_Audio")
OUT_DIR    = Path("/home/ke2461/MultiLLMDebate/BaselineDatasets")
OUT_DIR.mkdir(exist_ok=True)
SEED = 42

def extract_label_from_folder(folder_name):
    # Native English speakers: ENS_XX1, ENS_XX2, ENS_XX3
    if "ENS_XX" in folder_name:
        return "native"
    # Match A2, B1_1, B1_2, B2
    match = re.search(r'_(A2|B1_1|B1_2|B2)_', folder_name)
    if match:
        return match.group(1)
    return None

def find_audio(txt_filename):
    """
    txt filename: SM_CHN_PTJ1_056_A2_0.txt
    audio filename: SM_CHN_PTJ1_056_A2_0.mp3 (same name, different extension)
    """
    stem = Path(txt_filename).stem  # e.g. SM_CHN_PTJ1_056_A2_0
    for audio_folder in AUDIO_ROOT.iterdir():
        if not audio_folder.is_dir():
            continue
        mp3 = audio_folder / (stem + ".mp3")
        if mp3.exists():
            return str(mp3)
    return ""

records = []
skipped_folders = []

for folder in sorted(TEXT_ROOT.iterdir()):
    if not folder.is_dir():
        continue
    label = extract_label_from_folder(folder.name)
    if label is None:
        skipped_folders.append(folder.name)
        continue
    for txt_file in sorted(folder.glob("*.txt")):
        transcript = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
        if not transcript:
            continue
        audio_path = find_audio(txt_file.name)
        records.append({
            "speaker_id":  txt_file.stem,
            "cefr_label":  label,
            "transcript":  transcript,
            "audio_path":  audio_path,
        })

if skipped_folders:
    print(f"Skipped folders (no label match): {skipped_folders}")

df = pd.DataFrame(records)
print(f"\nTotal samples parsed: {len(df)}")
print(df["cefr_label"].value_counts())

# Split: match paper Table 1 (3898 train / 217 dev / 217 test)
train_df, temp_df = train_test_split(
    df, test_size=434, random_state=SEED, stratify=df["cefr_label"]
)
dev_df, test_df = train_test_split(
    temp_df, test_size=217, random_state=SEED, stratify=temp_df["cefr_label"]
)

print(f"\nTrain: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
print("\nTrain distribution:\n", train_df["cefr_label"].value_counts())
print("\nDev distribution:\n",   dev_df["cefr_label"].value_counts())
print("\nTest distribution:\n",  test_df["cefr_label"].value_counts())

train_df.to_csv(OUT_DIR / "train.csv", index=False)
dev_df.to_csv(OUT_DIR   / "dev.csv",   index=False)
test_df.to_csv(OUT_DIR  / "test.csv",  index=False)
print(f"\nSaved to {OUT_DIR}/")