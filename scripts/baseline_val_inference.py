# scripts/baseline_val_inference.py
"""
Runs inference on the validation set using already-trained baseline model checkpoints.
No retraining — loads best .pt checkpoint and evaluates on dev set only.
Saves results to logs/baseline_val/

For the comparison table with multi-LLM, also computes merged 4-label metrics
where B1_1 and B1_2 are both mapped to B1.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor,
)
from scipy.stats import pearsonr
from tqdm import tqdm

LOG_DIR  = Path("../logs")
OUT_DIR  = Path("../logs/baseline_val")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BERT_NAME     = "bert-base-uncased"
W2V_NAME      = "facebook/wav2vec2-base"
MAX_LEN       = 128
BATCH_SIZE    = 8
SAMPLE_RATE   = 16000
MAX_AUDIO_LEN = 16000 * 60
K_PROTO       = 3
SEED          = 42

# No merging needed — multi-LLM now uses same 5-label space as baseline
MERGED_LABELS   = ["A2", "B1_1", "B1_2", "B2", "native"]
LABEL2ID_MERGED = {l: i for i, l in enumerate(MERGED_LABELS)}
ID2LABEL_MERGED = {i: l for l, i in LABEL2ID_MERGED.items()}
B1_MERGE        = {}  # no merging — B1_1 and B1_2 kept separate


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def compute_metrics(y_true, y_pred, n_classes):
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)
    acc  = float(np.mean(y_true == y_pred))
    adj  = float(np.mean(np.abs(y_true - y_pred) <= 1))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pcc  = float(pearsonr(y_true, y_pred)[0]) if len(np.unique(y_true)) > 1 else float("nan")
    per_acc, per_adj, per_mse = [], [], []
    for c in range(n_classes):
        idx = np.where(y_true == c)[0]
        if len(idx) == 0:
            continue
        per_acc.append(float(np.mean(y_true[idx] == y_pred[idx])))
        per_adj.append(float(np.mean(np.abs(y_true[idx] - y_pred[idx]) <= 1)))
        per_mse.append(float(np.mean((y_true[idx] - y_pred[idx]) ** 2)))
    return {
        "ACC":     round(acc * 100, 2),
        "ADJ":     round(adj * 100, 2),
        "ACC_MC":  round(float(np.mean(per_acc)) * 100, 2),
        "ADJ_MC":  round(float(np.mean(per_adj)) * 100, 2),
        "RMSE":    round(rmse, 4),
        "RMSE_MC": round(float(np.sqrt(np.mean(per_mse))), 4),
        "PCC":     round(pcc, 4),
    }


# ── Text dataset (BERT) ───────────────────────────────────
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.texts  = df["transcript"].astype(str).tolist()
        self.labels = [label2id[l] for l in df["cefr_label"]]
        self.tok    = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx], padding="max_length",
            truncation=True, max_length=MAX_LEN, return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Audio dataset (W2V) ───────────────────────────────────
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, df, label2id):
        self.paths  = df["audio_path"].tolist()
        self.labels = [label2id[l] for l in df["cefr_label"]]
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.paths[idx])
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.mean(0)[:MAX_AUDIO_LEN]
        return {
            "waveform": waveform,
            "label":    torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_collate_fn(fe):
    def collate_fn(batch):
        waveforms = [b["waveform"].numpy() for b in batch]
        labels    = torch.stack([b["label"] for b in batch])
        encoded   = fe(waveforms, sampling_rate=SAMPLE_RATE,
                       return_tensors="pt", padding=True)
        return {"input_values": encoded.input_values, "labels": labels}
    return collate_fn


# ── Model architectures ───────────────────────────────────
class PrototypicalBERT(nn.Module):
    def __init__(self, num_classes, sim="COS"):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(BERT_NAME)
        self.sim        = sim
        self.prototypes = nn.Parameter(torch.randn(num_classes, K_PROTO, 768) * 0.01)
        if sim == "COS":
            self.s = nn.Parameter(torch.tensor(10.0))
            self.b = nn.Parameter(torch.tensor(0.0))
    def forward(self, input_ids, attention_mask):
        out   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        mask  = attention_mask.unsqueeze(-1).float()
        emb   = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        proto = self.prototypes.mean(dim=1)
        if self.sim == "COS":
            e_n    = nn.functional.normalize(emb,   dim=-1)
            p_n    = nn.functional.normalize(proto, dim=-1)
            scores = self.s * (e_n @ p_n.T) + self.b
        else:
            diff   = emb.unsqueeze(1) - proto.unsqueeze(0)
            scores = -(diff ** 2).sum(-1)
        return scores


class PrototypicalW2V(nn.Module):
    def __init__(self, num_classes, sim="COS"):
        super().__init__()
        self.w2v        = Wav2Vec2Model.from_pretrained(W2V_NAME)
        self.w2v.feature_extractor._freeze_parameters()
        self.sim        = sim
        self.prototypes = nn.Parameter(torch.randn(num_classes, K_PROTO, 768) * 0.01)
        if sim == "COS":
            self.s = nn.Parameter(torch.tensor(10.0))
            self.b = nn.Parameter(torch.tensor(0.0))
    def forward(self, input_values):
        out   = self.w2v(input_values).last_hidden_state
        emb   = out.mean(dim=1)
        proto = self.prototypes.mean(dim=1)
        if self.sim == "COS":
            e_n    = nn.functional.normalize(emb,   dim=-1)
            p_n    = nn.functional.normalize(proto, dim=-1)
            scores = self.s * (e_n @ p_n.T) + self.b
        else:
            diff   = emb.unsqueeze(1) - proto.unsqueeze(0)
            scores = -(diff ** 2).sum(-1)
        return scores


class W2VClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.w2v        = Wav2Vec2Model.from_pretrained(W2V_NAME)
        self.w2v.feature_extractor._freeze_parameters()
        self.classifier = nn.Linear(768, num_classes)
    def forward(self, input_values):
        out = self.w2v(input_values).last_hidden_state
        emb = out.mean(dim=1)
        return self.classifier(emb)


# ── Evaluation helpers ────────────────────────────────────
def eval_text(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            y_true += batch["labels"].tolist()
            ids    = batch["input_ids"].to(device)
            mask   = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask)
            if hasattr(logits, "logits"):
                logits = logits.logits
            y_pred += logits.argmax(-1).cpu().tolist()
    return y_true, y_pred


def eval_audio(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            y_true += batch["labels"].tolist()
            scores  = model(batch["input_values"].to(device))
            y_pred += scores.argmax(-1).cpu().tolist()
    return y_true, y_pred


def save_metrics(m, tag, filename):
    print(f"\n{'='*50}\n{tag} — VAL RESULTS\n{'='*50}")
    for k, v in m.items():
        print(f"  {k}: {v}")
    out_path = OUT_DIR / filename
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"  Saved → {out_path}")


def merge_predictions(y_true, y_pred, label2id_5):
    """No merging needed — both pipelines use the same 5-label space."""
    return y_true, y_pred


# ── Main ──────────────────────────────────────────────────
def run_all():
    set_seed(SEED)
    device = torch.device("cuda:1")
    print(f"Device: {device}")

    # Load val set CSVs
    dev_text_df  = pd.read_csv("../BaselineDatasets/dev_with_transcripts.csv")
    dev_audio_df = pd.read_csv("../BaselineDatasets/dev.csv")
    train_df     = pd.read_csv("../BaselineDatasets/train.csv")

    # Label mapping from training set (5 labels — must match checkpoints)
    labels_5   = sorted(train_df["cefr_label"].unique())
    label2id_5 = {l: i for i, l in enumerate(labels_5)}
    id2label_5 = {i: l for l, i in label2id_5.items()}
    n5         = len(labels_5)
    print(f"5-label set (baseline): {label2id_5}")

    # Filter dev sets to labels seen in training
    dev_text_df  = dev_text_df[dev_text_df["cefr_label"].isin(label2id_5)].reset_index(drop=True)
    dev_audio_df = dev_audio_df[
        dev_audio_df["cefr_label"].isin(label2id_5) &
        dev_audio_df["audio_path"].notna() &
        (dev_audio_df["audio_path"] != "")
    ].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
    fe        = Wav2Vec2FeatureExtractor.from_pretrained(W2V_NAME)
    collate   = make_collate_fn(fe)

    all_results        = {}   # 5-label metrics
    all_results_merged = {}   # 4-label merged metrics for comparison table

    def run_text_model(tag, file_tag, model):
        ckpt = LOG_DIR / f"{file_tag}_best.pt"
        if not ckpt.exists():
            print(f"  ⚠ Checkpoint not found: {ckpt}  — skipping {tag}")
            return
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device)
        loader = DataLoader(
            TextDataset(dev_text_df, tokenizer, label2id_5),
            batch_size=BATCH_SIZE
        )
        y_true, y_pred = eval_text(model, loader, device)

        # 5-label metrics
        m = compute_metrics(y_true, y_pred, n5)
        all_results[tag] = m
        save_metrics(m, tag, f"{file_tag}_val_metrics.json")

        # 4-label merged metrics
        y_true_m, y_pred_m = merge_predictions(y_true, y_pred, label2id_5)
        m_merged = compute_metrics(y_true_m, y_pred_m, len(MERGED_LABELS))
        all_results_merged[tag] = m_merged
        save_metrics(m_merged, f"{tag} (merged 4-label)", f"{file_tag}_val_merged_metrics.json")

    def run_audio_model(tag, file_tag, model):
        ckpt = LOG_DIR / f"{file_tag}_best.pt"
        if not ckpt.exists():
            print(f"  ⚠ Checkpoint not found: {ckpt}  — skipping {tag}")
            return
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device)
        loader = DataLoader(
            AudioDataset(dev_audio_df, label2id_5),
            batch_size=BATCH_SIZE, collate_fn=collate
        )
        y_true, y_pred = eval_audio(model, loader, device)

        # 5-label metrics
        m = compute_metrics(y_true, y_pred, n5)
        all_results[tag] = m
        save_metrics(m, tag, f"{file_tag}_val_metrics.json")

        # 4-label merged metrics
        y_true_m, y_pred_m = merge_predictions(y_true, y_pred, label2id_5)
        m_merged = compute_metrics(y_true_m, y_pred_m, len(MERGED_LABELS))
        all_results_merged[tag] = m_merged
        save_metrics(m_merged, f"{tag} (merged 4-label)", f"{file_tag}_val_merged_metrics.json")

    # ── BERT vanilla ──────────────────────────────────────
    run_text_model(
        "BERT", "bert",
        AutoModelForSequenceClassification.from_pretrained(
            BERT_NAME, num_labels=n5, id2label=id2label_5, label2id=label2id_5)
    )

    # ── BERT+LW ───────────────────────────────────────────
    run_text_model(
        "BERT+LW", "bert_lw",
        AutoModelForSequenceClassification.from_pretrained(
            BERT_NAME, num_labels=n5, id2label=id2label_5, label2id=label2id_5)
    )

    # ── PT-BERT variants ──────────────────────────────────
    for sim in ["COS", "SED"]:
        for lw in [False, True]:
            tag      = f"PT-BERT({sim})" + ("+LW" if lw else "")
            file_tag = f"bert_pt_{sim.lower()}" + ("_lw" if lw else "")
            run_text_model(tag, file_tag, PrototypicalBERT(num_classes=n5, sim=sim))

    # ── W2V vanilla ───────────────────────────────────────
    run_audio_model("W2V", "w2v", W2VClassifier(num_classes=n5))

    # ── W2V+LW ────────────────────────────────────────────
    run_audio_model("W2V+LW", "w2v_lw", W2VClassifier(num_classes=n5))

    # ── PT-W2V variants ───────────────────────────────────
    for sim in ["COS", "SED"]:
        for lw in [False, True]:
            tag      = f"PT-W2V({sim})" + ("+LW" if lw else "")
            file_tag = f"w2v_pt_{sim.lower()}" + ("_lw" if lw else "")
            run_audio_model(tag, file_tag, PrototypicalW2V(num_classes=n5, sim=sim))

    # ── Save combined summaries ───────────────────────────
    with open(OUT_DIR / "all_baseline_val_5label.json", "w") as f:
        json.dump(all_results, f, indent=2)
    with open(OUT_DIR / "all_baseline_val_4label_merged.json", "w") as f:
        json.dump(all_results_merged, f, indent=2)

    # ── Print comparison table (4-label merged) ───────────
    metrics_order = ["ACC", "ADJ", "RMSE", "PCC", "ACC_MC", "ADJ_MC", "RMSE_MC"]
    print(f"\n{'='*80}")
    print(f"  BASELINE VAL-SET COMPARISON TABLE (4-label merged — matches multi-LLM space)")
    print(f"{'─'*80}")
    print(f"  {'Model':<24}" + "".join(f"{m:>10}" for m in metrics_order))
    print(f"{'─'*80}")
    for model_name, m in all_results_merged.items():
        row = f"  {model_name:<24}" + "".join(f"{m.get(k, 0):>10}" for k in metrics_order)
        print(row)
    print(f"{'='*80}")
    print(f"\nAll results saved → {OUT_DIR}")


if __name__ == "__main__":
    run_all()