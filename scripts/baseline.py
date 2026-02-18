# baseline_bert_cefr.py

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import cross_entropy

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


# -----------------------------
# Paths / config
# -----------------------------
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)

BASELINE_PRED_CSV = LOG_DIR / "baseline_bert_predictions.csv"
BASELINE_METRICS_JSON = LOG_DIR / "baseline_bert_metrics.json"

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LR = 2e-5
WARMUP_RATIO = 0.1
SEED = 42


# -----------------------------
# Utilities: metrics
# -----------------------------
def compute_all_metrics(y_true_ids, y_pred_ids, id2label):
    """
    Compute ACC, ACC_MC, ADJ, ADJ_MC, RMSE, RMSE_MC, PCC
    similar to paper’s ACC, ACC_MC, RMSE, RMSE_MC + ADJ variants.
    Labels are treated as 0..K-1 ordinals.
    """
    y_true = np.array(y_true_ids, dtype=np.float32)
    y_pred = np.array(y_pred_ids, dtype=np.float32)

    # ---- Global ACC ----
    acc = np.mean(y_true == y_pred)

    # ---- Adjacent ACC: |pred - true| <= 1 ----
    adj = np.mean(np.abs(y_true - y_pred) <= 1)

    # ---- Global RMSE ----
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # ---- PCC ----
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        pcc, _ = pearsonr(y_true, y_pred)
    else:
        pcc = float("nan")

    # ---- Macro-type metrics ----
    n_classes = len(id2label)
    per_class_acc = []
    per_class_adj = []
    per_class_mse = []

    for c in range(n_classes):
        idx = np.where(y_true == c)[0]
        if len(idx) == 0:
            continue  # skip unseen class

        yt_c = y_true[idx]
        yp_c = y_pred[idx]

        per_class_acc.append(np.mean(yt_c == yp_c))
        per_class_adj.append(np.mean(np.abs(yt_c - yp_c) <= 1))
        per_class_mse.append(np.mean((yt_c - yp_c) ** 2))

    acc_mc = float(np.mean(per_class_acc)) if per_class_acc else float("nan")
    adj_mc = float(np.mean(per_class_adj)) if per_class_adj else float("nan")
    rmse_mc = float(np.sqrt(np.mean(per_class_mse))) if per_class_mse else float("nan")

    return {
        "ACC": float(acc),
        "ACC_MC": acc_mc,
        "ADJ": float(adj),
        "ADJ_MC": adj_mc,
        "RMSE": float(rmse),
        "RMSE_MC": rmse_mc,
        "PCC": float(pcc),
    }


# -----------------------------
# Dataset
# -----------------------------
class CEFRTextDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, text_col="transcript", label_col="cefr_label"):
        self.texts = df[text_col].astype(str).tolist()
        self.labels = [label2id[l] for l in df[label_col].tolist()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        label = self.labels[idx]
        enc = self.tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


# -----------------------------
# Training / evaluation helpers
# -----------------------------
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Train", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        labels = batch["labels"]

        if class_weights is not None:
            loss = cross_entropy(
                logits, labels, weight=class_weights.to(device)
            )
        else:
            loss = cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            labels = batch["labels"].numpy().tolist()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()

            all_labels.extend(labels)
            all_preds.extend(preds)

    return all_labels, all_preds


# -----------------------------
# Main: training baseline BERT
# -----------------------------
def run_baseline(
    train_csv="../train.csv",
    val_csv=None,
    test_csv=None,
    text_col="transcript",
    label_col="cefr_label",
):
    set_seed(SEED)

    # -------- Load data --------
    if val_csv is not None and test_csv is not None:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
    else:
        # Single CSV -> split into train/val/test (80/10/10) for convenience
        full_df = pd.read_csv(train_csv)
        train_df, temp_df = train_test_split(
            full_df, test_size=0.2, random_state=SEED, stratify=full_df[label_col]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=SEED, stratify=temp_df[label_col]
        )

    # -------- Label mapping (auto from data) --------
    labels_sorted = sorted(train_df[label_col].unique().tolist())
    label2id = {lab: i for i, lab in enumerate(labels_sorted)}
    id2label = {i: lab for lab, i in label2id.items()}
    num_labels = len(label2id)

    print("Label mapping:", label2id)

    # -------- Tokenizer & model --------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------- Datasets & loaders --------
    train_ds = CEFRTextDataset(train_df, tokenizer, label2id, text_col, label_col)
    val_ds = CEFRTextDataset(val_df, tokenizer, label2id, text_col, label_col)
    test_ds = CEFRTextDataset(test_df, tokenizer, label2id, text_col, label_col)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # -------- Class weights for imbalance (inverse frequency) --------
    class_counts = train_df[label_col].value_counts()
    freq = np.array([class_counts[id2label[i]] for i in range(num_labels)], dtype=np.float32)
    inv_freq = 1.0 / (freq + 1e-6)
    class_weights = torch.tensor(inv_freq / inv_freq.sum() * num_labels, dtype=torch.float32)

    # -------- Optimizer & scheduler --------
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    total_steps = NUM_EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # -------- Training loop --------
    best_val_rmse = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, class_weights
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validation
        val_true, val_pred = evaluate(model, val_loader, device)
        val_metrics = compute_all_metrics(val_true, val_pred, id2label)
        print("Validation metrics:", val_metrics)

        # Simple early "best" tracking by RMSE_MC
        if val_metrics["RMSE_MC"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE_MC"]
            torch.save(model.state_dict(), LOG_DIR / "baseline_bert_best.pt")
            print("  🔥 Saved new best model (by RMSE_MC).")

    # -------- Test evaluation (using best model) --------
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(LOG_DIR / "baseline_bert_best.pt", map_location=device))
    model.to(device)

    test_true, test_pred = evaluate(model, test_loader, device)
    test_metrics = compute_all_metrics(test_true, test_pred, id2label)

    print("\n========== BERT BASELINE – TEST METRICS ==========")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k}: {v}")

    # -------- Save predictions & metrics for comparison --------
    # Map back to string labels for logging
    test_true_labels = [id2label[i] for i in test_true]
    test_pred_labels = [id2label[i] for i in test_pred]

    out_df = pd.DataFrame({
        "text": test_df[text_col].tolist(),
        "true_label": test_true_labels,
        "pred_label": test_pred_labels,
    })
    out_df.to_csv(BASELINE_PRED_CSV, index=False)

    with open(BASELINE_METRICS_JSON, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nSaved prediction CSV to: {BASELINE_PRED_CSV}")
    print(f"Saved metrics JSON to:    {BASELINE_METRICS_JSON}")


if __name__ == "__main__":
    # Adjust these paths/columns to match your data.
    # If you only have ../train.csv, the script will auto-split into train/val/test.
    run_baseline(
        train_csv="../BaselineDatasets/train_with_transcripts.csv",
        val_csv="../BaselineDatasets/dev_with_transcripts.csv",
        test_csv="../BaselineDatasets/test_with_transcripts.csv",
        text_col="transcript",
        label_col="cefr_label",
    )


