# scripts/metrics.py
"""
Centralised metric computation and error analysis for the MultiLLMDebate project.
Implements all 7 metrics matching Table 1 of Banno & Matassoni (2022).
"""
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict

CEFR_LEVELS = ["A2", "B1", "B2", "native"]
CEFR_TO_NUM = {"A2": 1, "B1": 2, "B2": 3, "native": 4}
NUM_TO_CEFR = {v: k for k, v in CEFR_TO_NUM.items()}
VALID_CEFR  = set(CEFR_LEVELS)
CEFR_ORDER  = CEFR_LEVELS  # alias used by referee clamp logic


# ─────────────────────────────────────────────
# 7-Metric computation
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """
    Computes all 7 evaluation metrics matching Table 1 of Banno & Matassoni (2022).

    Parameters
    ----------
    y_true : list[int]   ordinal CEFR values (1–6)
    y_pred : list[int]   ordinal CEFR values (1–6)

    Returns
    -------
    dict with keys: ACC, ACC_ADJ, ACC_MC, ACC_MC_ADJ, RMSE, RMSE_MC, PCC
    """
    if len(y_true) == 0:
        print("[WARN] compute_metrics called with empty lists — returning zeros.")
        return {k: 0.0 for k in
                ["ACC", "ACC_ADJ", "ACC_MC", "ACC_MC_ADJ", "RMSE", "RMSE_MC", "PCC"]}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ── Standard (micro) metrics ──────────────────────────────────────────────
    acc  = float(np.mean(y_true == y_pred))
    adj  = float(np.mean(np.abs(y_true - y_pred) <= 1))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    pcc, _ = pearsonr(y_true, y_pred)

    # ── Macro (MC) metrics: per-class values averaged equally ────────────────
    # This prevents dominant classes from inflating overall numbers.
    class_correct     = defaultdict(int)
    class_adj_correct = defaultdict(int)
    class_total       = defaultdict(int)
    class_sq_errors   = defaultdict(list)

    for t, p in zip(y_true, y_pred):
        class_total[t]           += 1
        class_sq_errors[t].append((t - p) ** 2)
        if t == p:
            class_correct[t]     += 1
        if abs(t - p) <= 1:
            class_adj_correct[t] += 1

    classes_present = sorted(class_total.keys())

    acc_mc = float(np.mean([
        class_correct[c] / class_total[c] for c in classes_present
    ]))
    acc_mc_adj = float(np.mean([
        class_adj_correct[c] / class_total[c] for c in classes_present
    ]))
    rmse_mc = float(np.mean([
        np.sqrt(np.mean(class_sq_errors[c])) for c in classes_present
    ]))

    return {
        "ACC":        round(acc * 100,     2),   # reported as % to match paper
        "ACC_ADJ":    round(adj * 100,     2),
        "ACC_MC":     round(acc_mc * 100,  2),
        "ACC_MC_ADJ": round(acc_mc_adj * 100, 2),
        "RMSE":       round(rmse,          4),
        "RMSE_MC":    round(rmse_mc,       4),
        "PCC":        round(float(pcc),    4),
    }


def print_metrics_table(metrics, title=""):
    """Pretty-prints the 7-metric results as a compact table."""
    print(f"\n{'='*55}")
    if title:
        print(f"  {title}")
        print(f"{'─'*55}")
    print(f"  {'Metric':<14} {'Value':>10}")
    print(f"{'─'*55}")
    for k, v in metrics.items():
        unit = "%" if k.startswith("ACC") else ""
        print(f"  {k:<14} {v:>9.4f}{unit}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────
# Step-by-step error analysis
# ─────────────────────────────────────────────

def analyse_round1(r1_outputs):
    """
    Analyse Round 1 predictions for a single sample.
    Prints per-model labels and flags inter-model disagreement.
    Returns a dict summarising R1 state.
    """
    labels = {o["model"]: o["label"] for o in r1_outputs}
    unique = set(labels.values())

    print(f"    [R1] Individual predictions:")
    for model, label in labels.items():
        valid_flag = "" if label in VALID_CEFR else "  ⚠ INVALID LABEL"
        print(f"         {model:<16} → {label}{valid_flag}")

    if len(unique) == 1:
        print(f"    [R1] ✓ Full agreement on {list(unique)[0]}")
    elif len(unique) == len(labels):
        print(f"    [R1] ✗ Total disagreement — all {len(labels)} models differ: {labels}")
    else:
        print(f"    [R1] ~ Partial disagreement — {len(unique)} distinct labels: {sorted(unique)}")

    return {"labels": labels, "unique_count": len(unique)}


def analyse_round2(r1_outputs, r2_outputs):
    """
    Analyse Round 2 predictions and label shifts for a single sample.
    Prints which models changed their label and in which direction.
    Returns a dict summarising changes.
    """
    r1_labels = {o["model"]: o["label"]         for o in r1_outputs}
    r2_labels = {o["model"]: o["updated_label"] for o in r2_outputs}

    print(f"    [R2] Label updates after debate:")
    changes = {}
    for model in r2_labels:
        before = r1_labels.get(model, "?")
        after  = r2_labels[model]
        if before == after:
            print(f"         {model:<16} → {after}  (unchanged)")
        else:
            b_num = CEFR_TO_NUM.get(before, 0)
            a_num = CEFR_TO_NUM.get(after,  0)
            direction = "↑ RAISED" if a_num > b_num else "↓ LOWERED"
            print(f"         {model:<16} → {after}  ({before} → {after})  {direction}")
            changes[model] = (before, after)

    if not changes:
        print(f"    [R2] No models changed their label.")
    else:
        print(f"    [R2] {len(changes)}/{len(r2_labels)} model(s) changed label: "
              f"{[(m, f'{v[0]}→{v[1]}') for m, v in changes.items()]}")

    return {"r2_labels": r2_labels, "changes": changes}


def analyse_vote(votes, final_label):
    """
    Analyse the majority vote result for a single sample.
    Flags ties and vote distribution.
    """
    sorted_votes = sorted(votes.items(), key=lambda x: -x[1])
    print(f"    [VOTE] Distribution: {dict(sorted_votes)}  →  Final: {final_label}")

    top_count = sorted_votes[0][1]
    tied = [l for l, c in sorted_votes if c == top_count]
    if len(tied) > 1:
        print(f"    [VOTE] ⚠ TIE detected between: {tied} — first alphabetically wins: {sorted(tied)[0]}")


def analyse_referee(majority_label, ref_raw_label, clamped_label):
    """
    Analyse referee decision for a single sample.
    Shows whether referee agreed, changed, or was clamped.
    """
    print(f"    [REF]  Majority label : {majority_label}")
    print(f"    [REF]  Referee said   : {ref_raw_label}")
    print(f"    [REF]  After clamp    : {clamped_label}")

    if ref_raw_label == majority_label:
        print(f"    [REF]  ✓ Referee agreed with majority.")
    elif ref_raw_label == clamped_label:
        print(f"    [REF]  ~ Referee overrode majority (within ±1, clamp not triggered).")
    else:
        b_num = CEFR_TO_NUM.get(majority_label, 0)
        r_num = CEFR_TO_NUM.get(ref_raw_label,  0)
        c_num = CEFR_TO_NUM.get(clamped_label,  0)
        print(f"    [REF]  ✗ CLAMP TRIGGERED: referee wanted {ref_raw_label} "
              f"({abs(r_num - b_num)} levels from majority) — clamped to {clamped_label}.")


# ─────────────────────────────────────────────
# Post-hoc full error analysis
# ─────────────────────────────────────────────

def error_analysis_report(sample_details, pipeline_name="pipeline"):
    """
    Prints a comprehensive post-hoc error analysis across all samples.

    sample_details : list of dicts produced during the run, each containing:
        index, truth, pred, transcript_snippet,
        round1_labels (dict model→label),
        round2_labels (dict model→label),
        majority_pred (optional, for referee pipeline),
        ref_raw_label (optional, for referee pipeline),
        vote_distribution (dict)
    """
    print(f"\n{'#'*60}")
    print(f"  POST-HOC ERROR ANALYSIS  —  {pipeline_name}")
    print(f"{'#'*60}")

    total   = len(sample_details)
    errors  = [s for s in sample_details if s["truth"] != s["pred"]]
    correct = total - len(errors)

    print(f"\n  Samples total   : {total}")
    print(f"  Correct         : {correct}  ({100*correct/total:.1f}%)")
    print(f"  Errors          : {len(errors)}  ({100*len(errors)/total:.1f}%)")

    # ── Direction of errors ────────────────────────────────────────────────
    over = under = exact = 0
    for s in sample_details:
        diff = CEFR_TO_NUM.get(s["pred"], 0) - CEFR_TO_NUM.get(s["truth"], 0)
        if   diff > 0: over  += 1
        elif diff < 0: under += 1
        else:          exact += 1

    print(f"\n  Error direction:")
    print(f"    Exact match  : {exact}")
    print(f"    Over-rated   : {over}  (model predicted higher level than ground truth)")
    print(f"    Under-rated  : {under} (model predicted lower level than ground truth)")

    # ── Per-class accuracy ─────────────────────────────────────────────────
    print(f"\n  Per-class accuracy:")
    class_stats = defaultdict(lambda: {"correct": 0, "total": 0, "adj": 0})
    for s in sample_details:
        t = s["truth"]
        p = s["pred"]
        class_stats[t]["total"] += 1
        if t == p:
            class_stats[t]["correct"] += 1
        if abs(CEFR_TO_NUM.get(t, 0) - CEFR_TO_NUM.get(p, 0)) <= 1:
            class_stats[t]["adj"]     += 1

    for level in CEFR_LEVELS:
        if level in class_stats:
            c = class_stats[level]
            acc_pct = 100 * c["correct"] / c["total"] if c["total"] else 0
            adj_pct = 100 * c["adj"]     / c["total"] if c["total"] else 0
            print(f"    {level}:  {c['correct']:>3}/{c['total']:<3} exact ({acc_pct:5.1f}%)   "
                  f"adj: {c['adj']:>3}/{c['total']:<3} ({adj_pct:5.1f}%)")

    # ── Most common confusion pairs ────────────────────────────────────────
    print(f"\n  Most common error pairs (truth → pred):")
    pair_counts = defaultdict(int)
    for s in errors:
        pair_counts[(s["truth"], s["pred"])] += 1
    for (t, p), cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:10]:
        bar = "█" * cnt
        print(f"    {t} → {p}  :  {cnt:>3}x  {bar}")

    # ── R1 inter-model disagreement ────────────────────────────────────────
    print(f"\n  Round 1 disagreement analysis:")
    r1_full_agree = r1_partial = r1_full_disagree = 0
    for s in sample_details:
        r1 = s.get("round1_labels", {})
        n_unique = len(set(r1.values()))
        if n_unique == 1:             r1_full_agree    += 1
        elif n_unique == len(r1):     r1_full_disagree += 1
        else:                         r1_partial       += 1

    print(f"    Full agreement    : {r1_full_agree}")
    print(f"    Partial agreement : {r1_partial}")
    print(f"    Full disagreement : {r1_full_disagree}")

    # When R1 was fully agreed, how often was the final answer correct?
    agreed_correct = sum(
        1 for s in sample_details
        if len(set(s.get("round1_labels", {}).values())) == 1
        and s["truth"] == s["pred"]
    )
    agreed_total = r1_full_agree
    if agreed_total > 0:
        print(f"    Accuracy when R1 fully agreed: "
              f"{agreed_correct}/{agreed_total} = {100*agreed_correct/agreed_total:.1f}%")

    # ── R1 → R2 label shift analysis ──────────────────────────────────────
    print(f"\n  Round 1 → Round 2 label shift analysis:")
    total_positions = 0
    total_shifts    = 0
    shift_helped    = 0   # shifted AND final was correct (and R1 was wrong)
    shift_hurt      = 0   # shifted AND final was wrong (and R1 was correct)

    for s in sample_details:
        r1 = s.get("round1_labels", {})
        r2 = s.get("round2_labels", {})
        for model in r1:
            total_positions += 1
            if r1[model] != r2.get(model, r1[model]):
                total_shifts += 1
                if r1[model] == s["truth"] and r2[model] != s["truth"]:
                    shift_hurt  += 1
                elif r1[model] != s["truth"] and r2[model] == s["truth"]:
                    shift_helped += 1

    print(f"    Total model-positions : {total_positions}")
    print(f"    Label shifts (R1→R2)  : {total_shifts} ({100*total_shifts/max(total_positions,1):.1f}%)")
    print(f"    Shifts that helped    : {shift_helped}  (R1 wrong → R2 correct)")
    print(f"    Shifts that hurt      : {shift_hurt}   (R1 correct → R2 wrong)")

    # ── Referee-specific analysis (if applicable) ──────────────────────────
    if any("ref_raw_label" in s and s["ref_raw_label"] for s in sample_details):
        print(f"\n  Referee analysis:")
        ref_agree   = sum(1 for s in sample_details
                          if s.get("ref_raw_label") == s.get("majority_pred"))
        ref_clamped = sum(1 for s in sample_details
                          if s.get("ref_raw_label") != s.get("majority_pred")
                          and s.get("ref_raw_label") != s["pred"])
        ref_changed = sum(1 for s in sample_details
                          if s.get("majority_pred") != s["pred"])

        print(f"    Referee agreed with majority   : {ref_agree}/{total}")
        print(f"    Referee changed the decision   : {ref_changed}/{total}")
        print(f"    Clamp was triggered            : {ref_clamped}/{total}")

        # Of the times referee changed the result, how often was it correct?
        ref_changed_correct = sum(
            1 for s in sample_details
            if s.get("majority_pred") != s["pred"] and s["pred"] == s["truth"]
        )
        ref_changed_wrong = sum(
            1 for s in sample_details
            if s.get("majority_pred") != s["pred"] and s["pred"] != s["truth"]
        )
        print(f"    Of those changes: {ref_changed_correct} helped, {ref_changed_wrong} hurt")

    # ── UNK label analysis ────────────────────────────────────────────────
    unk_samples = [s for s in sample_details
                   if s["pred"] == "UNK" or any(
                       v == "UNK" for v in s.get("round1_labels", {}).values()
                   ) or any(
                       v == "UNK" for v in s.get("round2_labels", {}).values()
                   )]
    if unk_samples:
        print(f"\n  ⚠ UNK label appearances: {len(unk_samples)} samples affected")
        print(f"    (UNK means the model output could not be parsed as valid CEFR.)")
        print(f"    Affected indices: {[s['index'] for s in unk_samples[:10]]}")
    else:
        print(f"\n  ✓ No UNK labels — all outputs parsed successfully.")

    # ── Sample error cases ────────────────────────────────────────────────
    print(f"\n  Sample error cases (first 5):")
    for s in errors[:5]:
        print(f"\n    [Index {s['index']}]")
        print(f"    Truth: {s['truth']}  |  Pred: {s['pred']}")
        if s.get("round1_labels"):
            print(f"    R1 labels : {s['round1_labels']}")
        if s.get("round2_labels"):
            print(f"    R2 labels : {s['round2_labels']}")
        if s.get("majority_pred"):
            print(f"    Majority  : {s['majority_pred']}")
        if s.get("ref_raw_label"):
            print(f"    Referee   : {s['ref_raw_label']}")
        print(f"    Snippet   : {s.get('transcript_snippet', '')[:150]}...")

    print(f"\n{'#'*60}\n")