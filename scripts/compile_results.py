# compile_results.py
# Prints both validation-best and test results in Table 2 format

import json
from pathlib import Path

LOG_DIR = Path("../logs")

# Matches Table 2 row order exactly
experiments = [
    # (display name,           metrics json filename)
    ("BERT",                   "bert_metrics.json"),
    ("BERT +LW",               "bert_lw_metrics.json"),
    ("BERT PT(COS)",           "bert_pt_cos_metrics.json"),
    ("BERT PT(COS)+LW",        "bert_pt_cos_lw_metrics.json"),
    ("BERT PT(SED)",           "bert_pt_sed_metrics.json"),
    ("BERT PT(SED)+LW",        "bert_pt_sed_lw_metrics.json"),
    ("W2V",                    "w2v_metrics.json"),
    ("W2V +LW",                "w2v_lw_metrics.json"),
    ("W2V PT(COS)",            "w2v_pt_cos_metrics.json"),
    ("W2V PT(COS)+LW",         "w2v_pt_cos_lw_metrics.json"),
    ("W2V PT(SED)",            "w2v_pt_sed_metrics.json"),
    ("W2V PT(SED)+LW",         "w2v_pt_sed_lw_metrics.json"),
]

header = (f"{'Model':<20} {'RMSE':>6} {'RMSE_MC':>8} {'PCC':>6} "
          f"{'ACC':>7} {'ADJ':>7} {'ACC_MC':>8} {'ADJ_MC':>8}")
print(header)
print("-" * len(header))

for name, fname in experiments:
    p = LOG_DIR / fname
    if not p.exists():
        print(f"{name:<20}  — not yet run")
        continue
    m = json.load(open(p))
    print(f"{name:<20} "
          f"{m['RMSE']:>6.3f} {m['RMSE_MC']:>8.3f} {m['PCC']:>6.3f} "
          f"{m['ACC']*100:>7.2f} {m['ADJ']*100:>7.2f} "
          f"{m['ACC_MC']*100:>8.2f} {m['ADJ_MC']*100:>8.2f}")