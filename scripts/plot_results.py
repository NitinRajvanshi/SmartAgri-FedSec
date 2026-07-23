import os
import re
import matplotlib.pyplot as plt

from config import RESULTS_DIR

def parse_report(path):
    """
    Parse a sklearn classification_report text file.

    Returns:
        accuracy (float),
        weighted_precision (float),
        weighted_recall (float),
        weighted_f1 (float)
    """
    accuracy = None
    w_prec = w_rec = w_f1 = None

    if not os.path.exists(path):
        print(f"[WARN] Report not found: {path}")
        return None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # accuracy line: e.g. "accuracy                           0.5850     200"
            if line.startswith("accuracy"):
                # grab the first float in the line
                match = re.search(r"(\d+\.\d+)", line)
                if match:
                    accuracy = float(match.group(1))

            # weighted avg line: e.g.
            # "weighted avg       0.3422  0.5850  0.4318       200"
            if line.startswith("weighted avg"):
                parts = line.split()
                # parts = ["weighted", "avg", prec, rec, f1, support]
                # But sometimes "weighted avg" may be one token, so handle safely
                nums = [p for p in parts if re.match(r"^\d+\.\d+$", p)]
                if len(nums) >= 3:
                    w_prec = float(nums[0])
                    w_rec  = float(nums[1])
                    w_f1   = float(nums[2])

    if accuracy is None or w_prec is None:
        print(f"[WARN] Could not fully parse: {path}")
        return None

    return accuracy, w_prec, w_rec, w_f1

def main():
    methods = []
    accs    = []
    f1s     = []

    # 1) centralized
    c_path = os.path.join(RESULTS_DIR, "centralized_report.txt")
    c_metrics = parse_report(c_path)
    if c_metrics:
        a, p, r, f1 = c_metrics
        methods.append("Centralized")
        accs.append(a)
        f1s.append(f1)

    # 2) FL (no HE)
    fl_path = os.path.join(RESULTS_DIR, "federated_report.txt")
    fl_metrics = parse_report(fl_path)
    if fl_metrics:
        a, p, r, f1 = fl_metrics
        methods.append("FL")
        accs.append(a)
        f1s.append(f1)

    # 3) FL + HE
    flhe_path = os.path.join(RESULTS_DIR, "federated_he_report.txt")
    flhe_metrics = parse_report(flhe_path)
    if flhe_metrics:
        a, p, r, f1 = flhe_metrics
        methods.append("FL+HE")
        accs.append(a)
        f1s.append(f1)

    if not methods:
        print("No reports found in", RESULTS_DIR)
        return

    print("Methods:", methods)
    print("Accuracies:", accs)
    print("F1-scores:", f1s)

    # ---- Accuracy bar chart ----
    plt.figure()
    plt.bar(methods, [x * 100 for x in accs])
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy comparison of IDS methods")
    acc_path = os.path.join(RESULTS_DIR, "accuracy_comparison.png")
    plt.savefig(acc_path, bbox_inches="tight")
    print("Saved accuracy graph to:", acc_path)

    # ---- F1-score bar chart ----
    plt.figure()
    plt.bar(methods, [x * 100 for x in f1s])
    plt.ylabel("F1-score (%)")
    plt.title("F1-score comparison of IDS methods")
    f1_path = os.path.join(RESULTS_DIR, "f1_comparison.png")
    plt.savefig(f1_path, bbox_inches="tight")
    print("Saved F1-score graph to:", f1_path)

if __name__ == "__main__":
    main()
