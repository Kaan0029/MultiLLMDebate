# bert_classifier.py  —  BERT vanilla and BERT+LW
# Paper Section 4.1.1 + 4.3
# Usage:  python bert_classifier.py [--lw]

import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cross_entropy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "bert-base-uncased"
MAX_LEN    = 128
BATCH_SIZE = 8
LR         = 5e-5
PATIENCE   = 10
ALPHA      = 0.5
SEED       = 42

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

class CEFRDataset(Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.texts  = df["transcript"].astype(str).tolist()
        self.labels = [label2id[l] for l in df["cefr_label"]]
        self.tok    = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], padding="max_length",
                       truncation=True, max_length=MAX_LEN, return_tensors="pt")
        return {"input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

def make_weights_eq8(train_df, label2id, alpha=0.5):
    counts = train_df["cefr_label"].value_counts()
    n      = len(label2id)
    labs   = [None] * n
    for l, i in label2id.items(): labs[i] = l
    freq = np.array([counts[l] for l in labs], dtype=np.float32)
    p    = freq / freq.sum()
    p_a  = p ** alpha
    q    = (p_a / p_a.sum()) * (1.0 / p)
    return torch.tensor(q / q.sum() * n, dtype=torch.float32)

def train_epoch(model, loader, optimizer, device, weights=None):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        labels = batch["labels"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        optimizer.zero_grad()
        logits = model(input_ids=ids, attention_mask=mask).logits
        loss   = cross_entropy(logits, labels,
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
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            y_pred += logits.argmax(-1).cpu().tolist()
    return y_true, y_pred

def run(train_csv, dev_csv, test_csv, use_lw):
    set_seed(SEED)
    tag = "BERT+LW" if use_lw else "BERT"
    print(f"\n{'='*50}\nRunning: {tag}\n{'='*50}")

    train_df = pd.read_csv(train_csv)
    dev_df   = pd.read_csv(dev_csv)
    test_df  = pd.read_csv(test_csv)

    labels   = sorted(train_df["cefr_label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    print("Labels:", label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=len(labels),
                    id2label=id2label, label2id=label2id)
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(CEFRDataset(train_df, tokenizer, label2id),
                              batch_size=BATCH_SIZE, shuffle=True)
    dev_loader   = DataLoader(CEFRDataset(dev_df,   tokenizer, label2id),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(CEFRDataset(test_df,  tokenizer, label2id),
                              batch_size=BATCH_SIZE)

    weights     = make_weights_eq8(train_df, label2id, ALPHA) if use_lw else None
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR)
    file_tag    = "bert_lw" if use_lw else "bert"
    save_path   = LOG_DIR / f"{file_tag}_best.pt"
    best_acc_mc = -1.0
    patience_ct = 0
    epoch       = 0

    while patience_ct < PATIENCE:
        epoch += 1
        print(f"\n── Epoch {epoch} ({tag}) ──")
        loss = train_epoch(model, train_loader, optimizer, device, weights)
        print(f"Train loss: {loss:.4f}")
        y_true, y_pred = evaluate(model, dev_loader, device)
        m = compute_metrics(y_true, y_pred, id2label)
        print(f"Dev ACC_MC={m['ACC_MC']:.4f}  ACC={m['ACC']:.4f}  "
              f"ADJ={m['ADJ']:.4f}  RMSE={m['RMSE']:.4f}")
        if m["ACC_MC"] > best_acc_mc:
            best_acc_mc = m["ACC_MC"]; patience_ct = 0
            torch.save(model.state_dict(), save_path)
            print("  ✓ New best saved.")
        else:
            patience_ct += 1
            print(f"  No improvement ({patience_ct}/{PATIENCE})")

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