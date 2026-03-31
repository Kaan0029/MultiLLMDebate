# w2v_classifier.py  —  W2V vanilla and W2V+LW
# Paper Section 4.1.2 + 4.3
# Usage:  python w2v_classifier.py [--lw]

import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

MODEL_NAME    = "facebook/wav2vec2-base"
SAMPLE_RATE   = 16000
MAX_AUDIO_LEN = 16000 * 60   # cap at 60 seconds
BATCH_SIZE    = 8
LR            = 1e-5
EPOCHS        = 10
SEED          = 42

LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def compute_metrics(y_true, y_pred, id2label):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    acc  = float(np.mean(y_true == y_pred))
    adj  = float(np.mean(np.abs(y_true - y_pred) <= 1))
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    pcc  = float(pearsonr(y_true, y_pred)[0]) if len(np.unique(y_true)) > 1 else float("nan")
    n = len(id2label)
    per_acc, per_adj, per_mse = [], [], []
    for c in range(n):
        idx = np.where(y_true == c)[0]
        if len(idx) == 0: continue
        per_acc.append(np.mean(y_true[idx] == y_pred[idx]))
        per_adj.append(np.mean(np.abs(y_true[idx] - y_pred[idx]) <= 1))
        per_mse.append(np.mean((y_true[idx] - y_pred[idx])**2))
    return {"ACC": acc, "ACC_MC": float(np.mean(per_acc)),
            "ADJ": adj, "ADJ_MC": float(np.mean(per_adj)),
            "RMSE": rmse, "RMSE_MC": float(np.sqrt(np.mean(per_mse))),
            "PCC": pcc}

class AudioDataset(Dataset):
    """Returns raw waveform tensors — collation handles padding."""
    def __init__(self, df, label2id):
        self.paths  = df["audio_path"].tolist()
        self.labels = [label2id[l] for l in df["cefr_label"]]
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.paths[idx])
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(0)[:MAX_AUDIO_LEN]   # mono, cap length
        return {"waveform": waveform,
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}

def make_collate_fn(feature_extractor):
    """Pad variable-length waveforms using the HF feature extractor."""
    def collate_fn(batch):
        waveforms = [b["waveform"].numpy() for b in batch]
        labels    = torch.stack([b["label"] for b in batch])
        encoded   = feature_extractor(
            waveforms,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True       # pads all to the longest in the batch
        )
        return {"input_values": encoded.input_values, "labels": labels}
    return collate_fn

class W2VClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.w2v        = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        self.w2v.feature_extractor._freeze_parameters()
        self.classifier = nn.Linear(768, num_classes)
    def forward(self, input_values):
        out = self.w2v(input_values).last_hidden_state  # [B, T, 768]
        emb = out.mean(dim=1)                           # mean pool
        return self.classifier(emb)

def make_weights_eq9(train_df, label2id):
    """Paper Eq. 9: q̂_i = sum(p_j) / p_i  (simple inverse frequency for W2V)"""
    counts = train_df["cefr_label"].value_counts()
    n      = len(label2id)
    labs   = [None] * n
    for l, i in label2id.items(): labs[i] = l
    freq = np.array([counts[l] for l in labs], dtype=np.float32)
    p    = freq / freq.sum()
    q    = 1.0 / p
    return torch.tensor(q / q.sum() * n, dtype=torch.float32)

def train_epoch(model, loader, optimizer, device, weights=None):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        labels = batch["labels"].to(device)
        inputs = batch["input_values"].to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss   = nn.functional.cross_entropy(
                     logits, labels,
                     weight=weights.to(device) if weights is not None else None)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            y_true += batch["labels"].tolist()
            logits  = model(batch["input_values"].to(device))
            y_pred += logits.argmax(-1).cpu().tolist()
    return y_true, y_pred

def run(train_csv, dev_csv, test_csv, use_lw):
    set_seed(SEED)
    tag = "W2V+LW" if use_lw else "W2V"
    print(f"\n{'='*50}\nRunning: {tag}\n{'='*50}")

    train_df = pd.read_csv(train_csv)
    dev_df   = pd.read_csv(dev_csv)
    test_df  = pd.read_csv(test_csv)

    # Drop rows with missing audio
    for df, name in [(train_df,"train"),(dev_df,"dev"),(test_df,"test")]:
        n = df["audio_path"].isna().sum() + (df["audio_path"] == "").sum()
        if n: print(f"Warning: dropping {n} rows in {name} (no audio)")
    train_df = train_df[train_df["audio_path"].notna() & (train_df["audio_path"] != "")].reset_index(drop=True)
    dev_df   = dev_df  [dev_df  ["audio_path"].notna() & (dev_df  ["audio_path"] != "")].reset_index(drop=True)
    test_df  = test_df [test_df ["audio_path"].notna() & (test_df ["audio_path"] != "")].reset_index(drop=True)

    labels   = sorted(train_df["cefr_label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    print("Labels:", label2id)

    fe      = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = W2VClassifier(num_classes=len(labels)).to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR)
    weights = make_weights_eq9(train_df, label2id) if use_lw else None
    collate = make_collate_fn(fe)

    train_loader = DataLoader(AudioDataset(train_df, label2id),
                              batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate)
    dev_loader   = DataLoader(AudioDataset(dev_df,   label2id),
                              batch_size=BATCH_SIZE, collate_fn=collate)
    test_loader  = DataLoader(AudioDataset(test_df,  label2id),
                              batch_size=BATCH_SIZE, collate_fn=collate)

    file_tag    = "w2v_lw" if use_lw else "w2v"
    save_path   = LOG_DIR / f"{file_tag}_best.pt"
    best_acc_mc = -1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS} ({tag}) ──")
        loss = train_epoch(model, train_loader, opt, device, weights)
        print(f"Train loss: {loss:.4f}")
        y_true, y_pred = evaluate(model, dev_loader, device)
        m = compute_metrics(y_true, y_pred, id2label)
        print(f"Dev ACC_MC={m['ACC_MC']:.4f}  ACC={m['ACC']:.4f}  "
              f"ADJ={m['ADJ']:.4f}  RMSE={m['RMSE']:.4f}")
        if m["ACC_MC"] > best_acc_mc:
            best_acc_mc = m["ACC_MC"]
            torch.save(model.state_dict(), save_path)
            print("  ✓ New best saved.")

    print("\nLoading best model for test...")
    model.load_state_dict(torch.load(save_path, map_location=device,
                                     weights_only=True))
    y_true, y_pred = evaluate(model, test_loader, device)
    m = compute_metrics(y_true, y_pred, id2label)
    print(f"\n{'='*50}\n{tag} — TEST RESULTS\n{'='*50}")
    for k, v in m.items(): print(f"  {k}: {v:.4f}")
    with open(LOG_DIR / f"{file_tag}_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print(f"Saved → logs/{file_tag}_metrics.json")
    return m

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lw", action="store_true")
    args = parser.parse_args()
    run("../BaselineDatasets/train.csv",
        "../BaselineDatasets/dev.csv",
        "../BaselineDatasets/test.csv",
        use_lw=args.lw)