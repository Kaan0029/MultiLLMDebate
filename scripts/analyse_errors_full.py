# scripts/analyse_errors_full.py
"""
Comprehensive aggregate analysis of a completed debate pipeline log.
Covers ALL samples (correct and incorrect) to give a full picture
of model behaviour, error patterns, and where to improve the architecture.

Usage:
    python3 analyse_errors_full.py --log ../logs/gpt_only_val_only_logs.jsonl
    python3 analyse_errors_full.py --log ../logs/gpt_referee_val_only_logs.jsonl
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from metrics import CEFR_LEVELS, CEFR_TO_NUM


# ─────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────

def load_logs(log_path):
    records = []
    raw    = Path(log_path).read_text(encoding="utf-8")
    chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
    for chunk in chunks:
        try:
            records.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return records


# ─────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────

def pct(num, denom):
    return f"{100 * num / denom:.1f}%" if denom > 0 else "N/A"


def print_section(title, lines, out):
    out.append(f"\n{'='*60}")
    out.append(f"  {title}")
    out.append(f"{'─'*60}")
    for line in lines:
        out.append(line)


def analyse(records, out):

    total   = len(records)
    correct = sum(1 for r in records if r.get("ground_truth") == r.get("final_label"))
    errors  = total - correct

    # ── 1. Overall summary ────────────────────────────────────────────────
    print_section("1. OVERALL SUMMARY", [
        f"  Total samples    : {total}",
        f"  Correct          : {correct}  ({pct(correct, total)})",
        f"  Errors           : {errors}   ({pct(errors, total)})",
    ], out)

    # ── 2. Per-class accuracy ─────────────────────────────────────────────
    class_total   = defaultdict(int)
    class_correct = defaultdict(int)
    class_adj     = defaultdict(int)
    for r in records:
        t = r.get("ground_truth")
        p = r.get("final_label")
        if not t or not p:
            continue
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1
        if abs(CEFR_TO_NUM.get(t, 0) - CEFR_TO_NUM.get(p, 0)) <= 1:
            class_adj[t] += 1

    lines = [f"  {'Level':<8} {'Total':>7} {'Correct':>9} {'ACC':>8} {'ADJ±1':>8}"]
    lines.append(f"  {'─'*45}")
    for level in CEFR_LEVELS:
        if level in class_total:
            n = class_total[level]
            c = class_correct[level]
            a = class_adj[level]
            lines.append(f"  {level:<8} {n:>7} {c:>9} {pct(c,n):>8} {pct(a,n):>8}")
    print_section("2. PER-CLASS ACCURACY", lines, out)

    # ── 3. Confusion matrix (error pairs) ────────────────────────────────
    pair_counts = defaultdict(int)
    for r in records:
        t = r.get("ground_truth")
        p = r.get("final_label")
        if t and p and t != p:
            pair_counts[(t, p)] += 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])
    lines = [f"  {'Truth→Pred':<16} {'Count':>7} {'Bar'}"]
    lines.append(f"  {'─'*45}")
    for (t, p), cnt in sorted_pairs:
        bar = "█" * cnt
        lines.append(f"  {t}→{p:<12} {cnt:>7}  {bar}")
    print_section("3. CONFUSION PAIRS (truth → prediction)", lines, out)

    # ── 4. Error direction ────────────────────────────────────────────────
    over = under = 0
    for r in records:
        t = CEFR_TO_NUM.get(r.get("ground_truth"), 0)
        p = CEFR_TO_NUM.get(r.get("final_label"),  0)
        if p > t:   over  += 1
        elif p < t: under += 1

    print_section("4. ERROR DIRECTION", [
        f"  Over-rated  (pred > truth) : {over}   ({pct(over,  errors)}  of errors)",
        f"  Under-rated (pred < truth) : {under}  ({pct(under, errors)} of errors)",
    ], out)

    # ── 5. Where errors originate: R1 vs R2 ──────────────────────────────
    r1_caused   = 0   # all models wrong in R1, stayed wrong
    r2_caused   = 0   # R1 majority was correct but R2 changed to wrong
    r1_and_r2   = 0   # R1 wrong, R2 made it worse
    recovered   = 0   # R1 wrong but R2 corrected it (final correct)

    for r in records:
        truth = r.get("ground_truth")
        final = r.get("final_label")
        if not truth or not final:
            continue

        r1_labels = [o["label"]         for o in r.get("round1", [])]
        r2_labels = [o["updated_label"] for o in r.get("round2", [])]

        if not r1_labels or not r2_labels:
            continue

        # Majority vote of R1 labels
        r1_votes = defaultdict(int)
        for l in r1_labels:
            r1_votes[l] += 1
        r1_majority = sorted(r1_votes.items(), key=lambda x: (-x[1], x[0]))[0][0]

        r2_votes = defaultdict(int)
        for l in r2_labels:
            r2_votes[l] += 1
        r2_majority = sorted(r2_votes.items(), key=lambda x: (-x[1], x[0]))[0][0]

        r1_correct = (r1_majority == truth)
        r2_correct = (r2_majority == truth)
        final_correct = (final == truth)

        if not r1_correct and not final_correct:
            r1_caused += 1
        if r1_correct and not final_correct:
            r2_caused += 1
        if not r1_correct and final_correct:
            recovered += 1

    print_section("5. ERROR ORIGIN: WHERE DID IT GO WRONG?", [
        f"  Error already in R1, persisted to final : {r1_caused}  ({pct(r1_caused, errors)} of errors)",
        f"  R1 was correct but R2 introduced error  : {r2_caused}  ({pct(r2_caused, errors)} of errors)",
        f"  R1 was wrong but R2 recovered correctly : {recovered}",
        f"",
        f"  → If R1-caused errors dominate: improve Round 1 prompt or examples.",
        f"  → If R2-caused errors dominate: debate is hurting — make R2 more conservative.",
    ], out)

    # ── 6. Per-model R1 accuracy ──────────────────────────────────────────
    model_r1_correct = defaultdict(int)
    model_r1_total   = defaultdict(int)
    for r in records:
        truth = r.get("ground_truth")
        for o in r.get("round1", []):
            model_r1_total[o["model"]] += 1
            if o["label"] == truth:
                model_r1_correct[o["model"]] += 1

    lines = [f"  {'Model':<16} {'Correct':>9} {'Total':>7} {'ACC':>8}"]
    lines.append(f"  {'─'*45}")
    for model in sorted(model_r1_total.keys()):
        c = model_r1_correct[model]
        n = model_r1_total[model]
        lines.append(f"  {model:<16} {c:>9} {n:>7} {pct(c,n):>8}")
    print_section("6. PER-MODEL ACCURACY IN ROUND 1", lines, out)

    # ── 7. Per-model R2 accuracy ──────────────────────────────────────────
    model_r2_correct = defaultdict(int)
    model_r2_total   = defaultdict(int)
    for r in records:
        truth = r.get("ground_truth")
        for o in r.get("round2", []):
            model_r2_total[o["model"]] += 1
            if o["updated_label"] == truth:
                model_r2_correct[o["model"]] += 1

    lines = [f"  {'Model':<16} {'Correct':>9} {'Total':>7} {'ACC':>8}"]
    lines.append(f"  {'─'*45}")
    for model in sorted(model_r2_total.keys()):
        c = model_r2_correct[model]
        n = model_r2_total[model]
        lines.append(f"  {model:<16} {c:>9} {n:>7} {pct(c,n):>8}")
    print_section("7. PER-MODEL ACCURACY IN ROUND 2", lines, out)

    # ── 8. Label shift analysis (R1 → R2 per model) ───────────────────────
    model_shifts         = defaultdict(int)
    model_total_pos      = defaultdict(int)
    model_shift_helped   = defaultdict(int)   # was wrong in R1, correct in R2
    model_shift_hurt     = defaultdict(int)   # was correct in R1, wrong in R2

    for r in records:
        truth   = r.get("ground_truth")
        r1_dict = {o["model"]: o["label"]         for o in r.get("round1", [])}
        r2_dict = {o["model"]: o["updated_label"] for o in r.get("round2", [])}
        for model in r1_dict:
            model_total_pos[model] += 1
            before = r1_dict[model]
            after  = r2_dict.get(model, before)
            if before != after:
                model_shifts[model] += 1
                if before != truth and after == truth:
                    model_shift_helped[model] += 1
                elif before == truth and after != truth:
                    model_shift_hurt[model]   += 1

    lines = [f"  {'Model':<16} {'Shifts':>8} {'Total':>7} {'Shift%':>8} "
             f"{'Helped':>8} {'Hurt':>6}"]
    lines.append(f"  {'─'*60}")
    for model in sorted(model_total_pos.keys()):
        s = model_shifts[model]
        n = model_total_pos[model]
        h = model_shift_helped[model]
        b = model_shift_hurt[model]
        lines.append(f"  {model:<16} {s:>8} {n:>7} {pct(s,n):>8} "
                     f"{h:>8} {b:>6}")
    print_section("8. LABEL SHIFTS R1→R2 PER MODEL "
                  "(helped=R1 wrong→R2 correct, hurt=R1 correct→R2 wrong)", lines, out)

    # ── 9. Inter-model agreement in R1 ───────────────────────────────────
    full_agree    = 0
    partial_agree = 0
    full_disagree = 0
    agree_correct = 0

    for r in records:
        truth     = r.get("ground_truth")
        r1_labels = [o["label"] for o in r.get("round1", [])]
        n_unique  = len(set(r1_labels))
        if n_unique == 1:
            full_agree += 1
            if r1_labels[0] == truth:
                agree_correct += 1
        elif n_unique == len(r1_labels):
            full_disagree += 1
        else:
            partial_agree += 1

    print_section("9. INTER-MODEL AGREEMENT IN ROUND 1", [
        f"  Full agreement    : {full_agree}   ({pct(full_agree,    total)})",
        f"  Partial agreement : {partial_agree}  ({pct(partial_agree, total)})",
        f"  Full disagreement : {full_disagree}   ({pct(full_disagree, total)})",
        f"",
        f"  When R1 fully agreed, how often was it correct: "
        f"{agree_correct}/{full_agree} = {pct(agree_correct, full_agree)}",
        f"",
        f"  → High full-agreement + wrong = models share a systematic bias.",
        f"  → High disagreement = task is genuinely ambiguous at this level.",
    ], out)

    # ── 10. Referee analysis (if present) ────────────────────────────────
    has_referee = any("referee" in r for r in records)
    if has_referee:
        ref_total      = 0
        ref_agreed     = 0
        ref_changed    = 0
        ref_clamped    = 0
        ref_helped     = 0
        ref_hurt       = 0

        for r in records:
            ref = r.get("referee")
            if not ref:
                continue
            truth      = r.get("ground_truth")
            majority   = r.get("majority_label")
            final      = r.get("final_label")
            raw_label  = ref.get("raw_label")
            clamped    = ref.get("clamped_final_label")

            ref_total += 1
            if raw_label == majority:
                ref_agreed += 1
            if final != majority:
                ref_changed += 1
                if final == truth:
                    ref_helped += 1
                else:
                    ref_hurt   += 1
            if raw_label != clamped:
                ref_clamped += 1

        print_section("10. REFEREE ANALYSIS", [
            f"  Samples with referee       : {ref_total}",
            f"  Referee agreed with majority : {ref_agreed}  ({pct(ref_agreed, ref_total)})",
            f"  Referee changed decision   : {ref_changed}  ({pct(ref_changed, ref_total)})",
            f"  Clamp triggered            : {ref_clamped}  ({pct(ref_clamped, ref_total)})",
            f"  Of changes: helped={ref_helped}  hurt={ref_hurt}",
            f"",
            f"  → If hurt > helped: referee is net negative, consider removing.",
            f"  → If clamp fires often: referee is too aggressive, tighten prompt.",
        ], out)

    # ── 11. UNK label analysis ────────────────────────────────────────────
    unk_final  = sum(1 for r in records if r.get("final_label") == "UNK")
    unk_r1     = sum(
        1 for r in records
        for o in r.get("round1", [])
        if o["label"] == "UNK"
    )
    unk_r2     = sum(
        1 for r in records
        for o in r.get("round2", [])
        if o["updated_label"] == "UNK"
    )
    print_section("11. UNK / PARSE FAILURE ANALYSIS", [
        f"  UNK in R1 outputs  : {unk_r1}",
        f"  UNK in R2 outputs  : {unk_r2}",
        f"  UNK as final label : {unk_final}",
        f"",
        f"  → High UNK count = models not following JSON output format.",
        f"  → Fix: strengthen format instructions in prompts.py.",
    ], out)


# ─────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True,
                        help="Path to .jsonl log file")
    parser.add_argument("--out", default=None,
                        help="Optional output .txt file to save report")
    args = parser.parse_args()

    print(f"\nLoading: {args.log}")
    records = load_logs(args.log)
    print(f"Records loaded: {len(records)}\n")

    out = []
    out.append("FULL DEBATE PIPELINE ANALYSIS REPORT")
    out.append(f"Log: {args.log}")
    out.append(f"Samples: {len(records)}")

    analyse(records, out)

    report = "\n".join(out)
    print(report)

    log_path = Path(args.log)
    out_path = args.out or str(log_path.parent / (log_path.stem + "_full_analysis.txt"))
    Path(out_path).write_text(report, encoding="utf-8")
    print(f"\n\nReport saved → {out_path}")


if __name__ == "__main__":
    main()