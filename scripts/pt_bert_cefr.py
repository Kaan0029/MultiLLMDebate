# pt_bert_cefr.py
# Metric-learning PT-BERT (TEXT) for CEFR prediction

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cosine_similarity

from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-5
SEED = 42

LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
METRICS_OUT = LOG_DIR / "pt_bert_metrics.json"


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Metrics (same as paper)
# -----------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    acc = np.mean(y_true == y_pred)
    adj = np.mean(np.abs(y_true - y_pred) <= 1)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    pcc, _ = pearsonr(y_true, y_pred) if len(np.unique(y_true)) > 1 else (np.nan, None)

    return {
        "ACC": float(acc),
        "ADJ": float(adj),
        "RMSE": float(rmse),
        "PCC": float(pcc),
    }


# -----------------------------
# Dataset
# -----------------------------
class CEFRDataset(Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.texts = df["transcript"].astype(str).tolist()
        self.labels = [label2id[l] for l in df["cefr_label"]]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# -----------------------------
# PT-BERT Model
# -----------------------------
class PTBERT(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        return cls_emb



# -----------------------------
# Prototype computation
# -----------------------------
def compute_prototypes(model, loader, device, num_classes):
    model.eval()
    reps = [[] for _ in range(num_classes)]

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].to(device)
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            emb = model(**batch)

            for e, l in zip(emb, labels):
                reps[l.item()].append(e.cpu())

    prototypes = [torch.stack(r).mean(dim=0) for r in reps]
    return torch.stack(prototypes).to(device)


# -----------------------------
# Training loop
# -----------------------------
def train_epoch(model, loader, optimizer, device, prototypes):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        labels = batch["labels"].to(device)
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}

        emb = model(**batch)
        dists = torch.cdist(emb, prototypes)  # Euclidean
        loss = nn.CrossEntropyLoss()( -dists, labels )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate(model, loader, prototypes, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].tolist()
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            emb = model(**batch)
            dists = torch.cdist(emb, prototypes)
            preds = torch.argmin(dists, dim=1).cpu().tolist()

            y_true.extend(labels)
            y_pred.extend(preds)

    return y_true, y_pred


# -----------------------------
# Main
# -----------------------------
def run_pt_bert(train_csv, dev_csv, test_csv):
    set_seed(SEED)

    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)
    test_df = pd.read_csv(test_csv)

    labels = sorted(train_df["cefr_label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PTBERT(MODEL_NAME).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_loader = DataLoader(
        CEFRDataset(train_df, tokenizer, label2id),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    dev_loader = DataLoader(
        CEFRDataset(dev_df, tokenizer, label2id),
        batch_size=BATCH_SIZE,
    )
    test_loader = DataLoader(
        CEFRDataset(test_df, tokenizer, label2id),
        batch_size=BATCH_SIZE,
    )

    # Training
    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        prototypes = compute_prototypes(model, train_loader, device, len(labels))
        loss = train_epoch(model, train_loader, optimizer, device, prototypes)
        print(f"Train loss: {loss:.4f}")

    # Final prototypes
    prototypes = compute_prototypes(model, train_loader, device, len(labels))

    # Test
    y_true, y_pred = evaluate(model, test_loader, prototypes, device)
    metrics = compute_metrics(y_true, y_pred)

    print("\n===== PT-BERT TEST METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    run_pt_bert(
        train_csv="../BaselineDatasets/train_with_transcripts.csv",
        dev_csv="../BaselineDatasets/dev_with_transcripts.csv",
        test_csv="../BaselineDatasets/test_with_transcripts.csv",
    )
