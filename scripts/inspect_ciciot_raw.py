# inspect_ciciot_raw.py
import os, glob, pandas as pd
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR_CANDS = [
    os.path.join(BASE, "data", "CICIoT2023_raw"),
    os.path.join(BASE, "data", "CICIoT2023"),
    os.path.join(BASE, "data", "ciciot2023"),
]

raw_dir = None
for d in RAW_DIR_CANDS:
    if os.path.isdir(d):
        raw_dir = d
        break

if raw_dir is None:
    raise FileNotFoundError("Raw CICIoT folder not found. Put CSVs into data/CICIoT2023_raw/")

print("Inspecting raw files in:", raw_dir)
csvs = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
print("Found", len(csvs), "csv files. Listing first 20 if many...\n")

for p in csvs[:50]:
    print("----", os.path.basename(p))
    try:
        df = pd.read_csv(p, nrows=200, low_memory=False)  # quick preview
    except Exception as e:
        print("  [ERROR reading file]:", e)
        continue

    # Attempt to find label column name
    cols_lc = {c.lower(): c for c in df.columns}
    label_col = None
    for key in ("label","attack","class","category"):
        if key in cols_lc:
            label_col = cols_lc[key]
            break
    if label_col is None:
        label_col = df.columns[-1]

    # Print sample values and counts (limited)
    vals = df[label_col].astype(str).str.strip().unique().tolist()[:10]
    print("  label_col:", label_col)
    print("  sample label values:", vals)

    try:
        # Count full file (may be slower) — do full count but graceful
        full_df = pd.read_csv(p, usecols=[label_col], low_memory=False)
        counts = Counter(full_df[label_col].astype(str).str.lower().str.strip().values)
        print("  full counts (top 10):", counts.most_common(10))
    except Exception as e:
        print("  [WARN] could not fully count labels due to size:", e)

    print()
