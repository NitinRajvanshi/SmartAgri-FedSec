# make_metrics_and_plots.py
"""
Parse sklearn classification_report text files, collect metrics, save CSV/MD and plot bar charts.

Place this file in your project root and run:
    python make_metrics_and_plots.py

It expects report files in RESULTS_DIR defined in config.py:
 - results/centralized_report.txt
 - results/federated_report.txt
 - results/federated_he_report.txt

Outputs:
 - results/metrics_table.csv
 - results/metrics_table.md
 - results/accuracy_comparison.png
 - results/f1_comparison.png
"""

import os
import re
import csv
import textwrap
import matplotlib.pyplot as plt

# If your code defines RESULTS_DIR in config.py, import it; otherwise use ./results
try:
    from config import RESULTS_DIR
except Exception:
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

REPORT_FILES = {
    "Centralized": os.path.join(RESULTS_DIR, "centralized_report.txt"),
    "FL": os.path.join(RESULTS_DIR, "federated_report.txt"),
    "FL+HE": os.path.join(RESULTS_DIR, "federated_he_report.txt"),
}

OUT_CSV = os.path.join(RESULTS_DIR, "metrics_table.csv")
OUT_MD  = os.path.join(RESULTS_DIR, "metrics_table.md")
OUT_ACC = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
OUT_F1  = os.path.join(RESULTS_DIR, "f1_comparison.png")

FLOAT_RE = re.compile(r"(\d+\.\d+)")

def parse_report(path):
    """
    Parse a classification_report .txt (sklearn style) and return:
    (accuracy, weighted_precision, weighted_recall, weighted_f1) as floats or None
    """
    if not os.path.exists(path):
        return None

    accuracy = None
    w_prec = w_rec = w_f1 = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]

    for ln in lines:
        # accuracy line often starts with 'accuracy'
        if ln.startswith("accuracy"):
            m = FLOAT_RE.search(ln)
            if m:
                accuracy = float(m.group(1))

        # weighted avg line may be 'weighted avg' or 'weighted_avg' depending on formatting
        if ln.lower().startswith("weighted avg") or ln.lower().startswith("weighted_avg") or ln.lower().startswith("weighted"):
            # pick floats from the line; first is precision, second recall, third f1
            nums = FLOAT_RE.findall(ln)
            if len(nums) >= 3:
                try:
                    w_prec, w_rec, w_f1 = map(float, nums[:3])
                except:
                    pass

    # fallback: try to find 'accuracy' value elsewhere (single float on line)
    if accuracy is None:
        for ln in lines[::-1]:
            if ln.startswith("accuracy"):
                m = FLOAT_RE.search(ln)
                if m:
                    accuracy = float(m.group(1))
                    break

    if accuracy is None and w_f1 is None:
        # couldn't parse
        return None

    return (accuracy, w_prec, w_rec, w_f1)

def collect_all():
    results = []
    for method, path in REPORT_FILES.items():
        parsed = parse_report(path)
        if parsed is None:
            print(f"[WARN] Could not parse report for {method}. File missing or format unexpected: {path}")
            continue
        accuracy, w_prec, w_rec, w_f1 = parsed
        results.append({
            "method": method,
            "accuracy": accuracy,
            "precision_weighted": w_prec,
            "recall_weighted": w_rec,
            "f1_weighted": w_f1,
            "report_path": path
        })
    return results

def save_csv(results, out_csv=OUT_CSV):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "report_path"])
        for r in results:
            writer.writerow([
                r["method"],
                r["accuracy"] if r["accuracy"] is not None else "",
                r["precision_weighted"] if r["precision_weighted"] is not None else "",
                r["recall_weighted"] if r["recall_weighted"] is not None else "",
                r["f1_weighted"] if r["f1_weighted"] is not None else "",
                r["report_path"]
            ])
    print("Saved metrics CSV to:", out_csv)

def save_markdown(results, out_md=OUT_MD):
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    header = "| Method | Accuracy (%) | Precision (weighted) | Recall (weighted) | F1 (weighted) |\n"
    header += "|---|---:|---:|---:|---:|\n"
    lines = [header]
    for r in results:
        acc = f"{r['accuracy']*100:.2f}%" if r['accuracy'] is not None else "N/A"
        p = f"{r['precision_weighted']*100:.2f}%" if r['precision_weighted'] is not None else "N/A"
        rec = f"{r['recall_weighted']*100:.2f}%" if r['recall_weighted'] is not None else "N/A"
        f1 = f"{r['f1_weighted']*100:.2f}%" if r['f1_weighted'] is not None else "N/A"
        lines.append(f"| {r['method']} | {acc} | {p} | {rec} | {f1} |\n")
    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Saved markdown metrics table to:", out_md)

def plot_bars(results, out_acc=OUT_ACC, out_f1=OUT_F1):
    if not results:
        print("[WARN] No results to plot.")
        return
    methods = [r["method"] for r in results]
    accs = [(r["accuracy"]*100 if r["accuracy"] is not None else 0.0) for r in results]
    f1s  = [(r["f1_weighted"]*100 if r["f1_weighted"] is not None else 0.0) for r in results]

    # Accuracy plot
    plt.figure(figsize=(6,4))
    bars = plt.bar(methods, accs)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy comparison")
    for bar, val in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_acc, dpi=200)
    plt.close()
    print("Saved accuracy chart to:", out_acc)

    # F1 plot
    plt.figure(figsize=(6,4))
    bars = plt.bar(methods, f1s)
    plt.ylim(0, 100)
    plt.ylabel("F1-score (weighted) (%)")
    plt.title("F1-score comparison")
    for bar, val in zip(bars, f1s):
        plt.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_f1, dpi=200)
    plt.close()
    print("Saved F1 chart to:", out_f1)

def print_markdown_table(results):
    # print table for quick copy-paste
    print("\n--- Metrics Table (Markdown) ---\n")
    print("| Method | Accuracy (%) | Precision (weighted) | Recall (weighted) | F1 (weighted) |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        acc = f"{r['accuracy']*100:.2f}" if r['accuracy'] is not None else "N/A"
        p   = f"{r['precision_weighted']*100:.2f}" if r['precision_weighted'] is not None else "N/A"
        rec = f"{r['recall_weighted']*100:.2f}" if r['recall_weighted'] is not None else "N/A"
        f1  = f"{r['f1_weighted']*100:.2f}" if r['f1_weighted'] is not None else "N/A"
        print(f"| {r['method']} | {acc}% | {p}% | {rec}% | {f1}% |")
    print("\n--- End of table ---\n")

def main():
    results = collect_all()
    if not results:
        print("No metrics parsed. Make sure the report files exist in:", RESULTS_DIR)
        return

    # keep the canonical order if present
    order = ["Centralized", "FL", "FL+HE"]
    results_sorted = sorted(results, key=lambda x: order.index(x["method"]) if x["method"] in order else 99)

    save_csv(results_sorted)
    save_markdown(results_sorted)
    plot_bars(results_sorted)
    print_markdown_table(results_sorted)

    # optionally, provide path to project pdf if you want to attach it in report (local path)
    # Replace or adjust if your PDF path differs.
    PROJECT_PDF = "/mnt/data/CYBERSECURITY PROJECT .pdf"
    if os.path.exists(PROJECT_PDF):
        print("Project PDF available at (local path):", PROJECT_PDF)
    else:
        # also show sandbox style path (some systems transform it)
        print("If you need to attach your project PDF, its known upload path was: sandbox:/mnt/data/CYBERSECURITY PROJECT .pdf")

if __name__ == "__main__":
    main()
