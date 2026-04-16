# scripts/inspect_errors.py
"""
Reads a completed debate pipeline log (.jsonl) and prints detailed
inspection of error cases — showing exactly where in the reasoning
chain the wrong label was introduced or propagated.

Usage:
    python3 inspect_errors.py --log ../logs/gpt_only_val_only_logs.jsonl --n 10
    python3 inspect_errors.py --log ../logs/gpt_referee_val_only_logs.jsonl --n 10
"""
import json
import argparse
from pathlib import Path

CEFR_TO_NUM = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

def load_logs(log_path):
    """Load all samples from a .jsonl log file."""
    records = []
    raw = Path(log_path).read_text(encoding="utf-8")
    # Each record is a JSON object separated by blank lines
    chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
    for chunk in chunks:
        try:
            records.append(json.loads(chunk))
        except json.JSONDecodeError:
            continue
    return records


def classify_error(truth, pred):
    """Describe the direction and magnitude of the error."""
    t = CEFR_TO_NUM.get(truth, 0)
    p = CEFR_TO_NUM.get(pred, 0)
    diff = p - t
    if diff == 0:
        return "CORRECT"
    direction = "OVER-RATED" if diff > 0 else "UNDER-RATED"
    return f"{direction} by {abs(diff)} level(s)  ({truth} → {pred})"


def find_error_stage(record, truth):
    """
    Traces through R1 → R2 → vote → referee (if present) to find
    where the error first appeared or was reinforced.
    Returns a string describing the stage.
    """
    r1_labels = {o["model"]: o["label"] for o in record.get("round1", [])}
    r2_labels = {o["model"]: o["updated_label"] for o in record.get("round2", [])}
    final     = record.get("final_label", "?")
    majority  = record.get("majority_label") or record.get("final_label", "?")
    referee   = record.get("referee", {})

    stages = []

    # ── R1 analysis ───────────────────────────────────────────────────────
    r1_correct   = {m: l for m, l in r1_labels.items() if l == truth}
    r1_incorrect = {m: l for m, l in r1_labels.items() if l != truth}

    if not r1_incorrect:
        stages.append("R1: All models correct ✓")
    elif not r1_correct:
        stages.append(f"R1: All models wrong ✗  — labels: {r1_labels}")
    else:
        stages.append(f"R1: Mixed — correct: {r1_correct}  wrong: {r1_incorrect}")

    # ── R2 analysis ───────────────────────────────────────────────────────
    r2_correct   = {m: l for m, l in r2_labels.items() if l == truth}
    r2_incorrect = {m: l for m, l in r2_labels.items() if l != truth}

    # Which models changed their label R1→R2?
    for model in r1_labels:
        before = r1_labels.get(model)
        after  = r2_labels.get(model)
        if before != after:
            direction = "improved ✓" if after == truth else "worsened ✗"
            stages.append(f"R2: {model} changed {before}→{after} ({direction})")

    if not r2_incorrect:
        stages.append("R2: All models correct after debate ✓")
    elif not r2_correct:
        stages.append(f"R2: All models still wrong ✗  — labels: {r2_labels}")

    # ── Vote analysis ──────────────────────────────────────────────────────
    votes = record.get("majority_vote", {})
    stages.append(f"VOTE: distribution={votes}  majority={majority}")
    if majority == truth:
        stages.append("VOTE: Majority was correct ✓")
    else:
        stages.append(f"VOTE: Majority wrong ✗  (truth={truth})")

    # ── Referee analysis (if present) ─────────────────────────────────────
    if referee:
        ref_raw   = referee.get("raw_label", "?")
        ref_final = referee.get("clamped_final_label", "?")
        if ref_final == truth:
            stages.append(f"REFEREE: Corrected to {ref_final} ✓")
        elif majority == truth and ref_final != truth:
            stages.append(f"REFEREE: Introduced error ✗  "
                          f"({majority}→{ref_final}, truth={truth})")
        else:
            stages.append(f"REFEREE: Did not fix error "
                          f"(said {ref_raw}, clamped to {ref_final})")

    return stages


def format_reasoning(record, truth):
    """Format the full reasoning chain for a single error sample."""
    lines = []
    sep = "─" * 60

    lines.append(sep)
    lines.append(f"  Sample index : {record.get('sample_index', '?')}")
    lines.append(f"  Ground truth : {truth}")
    lines.append(f"  Final pred   : {record.get('final_label', '?')}")
    lines.append(f"  Error type   : {classify_error(truth, record.get('final_label', '?'))}")
    lines.append("")

    # ── Round 1 reasoning ─────────────────────────────────────────────────
    lines.append("  [ ROUND 1 — Independent judgments ]")
    for o in record.get("round1", []):
        model  = o["model"]
        label  = o["label"]
        reason = o.get("parsed", {}).get("reason", o.get("raw", "")[:200])
        correct_str = "✓" if label == truth else "✗"
        lines.append(f"    {model:<16} → {label} {correct_str}")
        lines.append(f"    Reasoning: {str(reason)[:300]}")
        lines.append("")

    # ── Round 2 reasoning ─────────────────────────────────────────────────
    lines.append("  [ ROUND 2 — After seeing other models ]")
    r1_labels = {o["model"]: o["label"] for o in record.get("round1", [])}
    for o in record.get("round2", []):
        model         = o["model"]
        updated_label = o["updated_label"]
        reason        = o.get("parsed", {}).get("reason", o.get("raw", "")[:200])
        before        = r1_labels.get(model, "?")
        changed_str   = f"(changed from {before})" if before != updated_label else "(unchanged)"
        correct_str   = "✓" if updated_label == truth else "✗"
        lines.append(f"    {model:<16} → {updated_label} {correct_str}  {changed_str}")
        lines.append(f"    Reasoning: {str(reason)[:300]}")
        lines.append("")

    # ── Vote ──────────────────────────────────────────────────────────────
    votes    = record.get("majority_vote", {})
    majority = record.get("majority_label") or record.get("final_label", "?")
    lines.append(f"  [ VOTE ]  {votes}  →  majority = {majority} "
                 f"{'✓' if majority == truth else '✗'}")
    lines.append("")

    # ── Referee (if present) ──────────────────────────────────────────────
    referee = record.get("referee")
    if referee:
        lines.append("  [ REFEREE ]")
        lines.append(f"    Raw label     : {referee.get('raw_label', '?')}")
        lines.append(f"    Clamped final : {referee.get('clamped_final_label', '?')}")
        ref_reason = referee.get("parsed", {}).get("reason", "")
        if ref_reason:
            lines.append(f"    Reasoning: {str(ref_reason)[:300]}")
        lines.append("")

    # ── Stage-by-stage error trace ────────────────────────────────────────
    lines.append("  [ ERROR TRACE ]")
    for stage in find_error_stage(record, truth):
        lines.append(f"    {stage}")
    lines.append(sep)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True,
                        help="Path to .jsonl log file")
    parser.add_argument("--n",   type=int, default=10,
                        help="Number of error cases to inspect")
    parser.add_argument("--out", default=None,
                        help="Optional output .txt file path")
    parser.add_argument("--stratified", action="store_true",
                        help="Sample errors evenly across error types rather than taking first n")
    args = parser.parse_args()

    print(f"\nLoading logs from: {args.log}")
    records = load_logs(args.log)
    print(f"Total records loaded: {len(records)}")

    # Find error cases
    errors = [
        r for r in records
        if r.get("ground_truth") and r.get("final_label")
        and r["ground_truth"] != r["final_label"]
    ]
    print(f"Total errors found : {len(errors)}")

    # ── Select which errors to inspect ────────────────────────────────────
    if args.stratified:
        # Group by (truth, pred) pair and sample evenly across all error types
        # This ensures we see a representative spread rather than just the
        # most common error type repeated 10 times
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in errors:
            key = (r["ground_truth"], r["final_label"])
            buckets[key].append(r)
        sampled = []
        bucket_lists = list(buckets.values())
        i = 0
        while len(sampled) < args.n and any(bucket_lists):
            bucket = bucket_lists[i % len(bucket_lists)]
            if bucket:
                sampled.append(bucket.pop(0))
            i += 1
        errors_to_inspect = sampled
        print(f"Stratified sample  : {len(errors_to_inspect)} errors across "
              f"{len(set((r['ground_truth'], r['final_label']) for r in errors_to_inspect))} error types\n")
    else:
        errors_to_inspect = errors[:args.n]
        print(f"Inspecting first   : {min(args.n, len(errors))} error cases\n")

    output_lines = []
    output_lines.append(f"ERROR INSPECTION REPORT")
    output_lines.append(f"Log file : {args.log}")
    output_lines.append(f"Total errors : {len(errors)}  |  Showing : {len(errors_to_inspect)}")
    output_lines.append("=" * 60)

    for record in errors_to_inspect:
        truth     = record["ground_truth"]
        formatted = format_reasoning(record, truth)
        output_lines.append(formatted)
        print(formatted)

    # Save to file
    log_path = Path(args.log)
    suffix   = "_stratified_inspection.txt" if args.stratified else "_error_inspection.txt"
    out_path = args.out or str(log_path.parent / (log_path.stem + suffix))
    Path(out_path).write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\nFull report saved → {out_path}")


if __name__ == "__main__":
    main()