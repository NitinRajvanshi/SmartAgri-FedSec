# prepare_ciciot2023.py  (overwrite your previous version)
import os, glob, zipfile, re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# possible raw directories (from your inspect output it is data/CICIoT2023_raw)
POSSIBLE_RAW_DIRS = [
    os.path.join(BASE_DIR, "data", "CICIoT2023_raw"),
    os.path.join(BASE_DIR, "data", "CICIoT2023"),
    os.path.join(BASE_DIR, "data", "ciciot2023"),
]
OUT_PATH = os.path.join(BASE_DIR, "data", "iot_data_ciciot2023.csv")

# If you want a smaller balanced sample for quick experiments, set MAX_ROWS_PER_CLASS
# Set to None to keep all rows (may be huge and slow).
MAX_ROWS_PER_CLASS = 20000   # keep up to 20k benign + 20k attack (adjust to your machine)

def find_raw_dir():
    for d in POSSIBLE_RAW_DIRS:
        if os.path.isdir(d):
            return d
    # try to auto-extract a zip if present
    cand_zip = os.path.join(BASE_DIR, "data", "CICIoT2023.zip")
    if os.path.exists(cand_zip):
        out = os.path.join(BASE_DIR, "data", "CICIoT2023_raw")
        os.makedirs(out, exist_ok=True)
        print("Extracting zip:", cand_zip)
        with zipfile.ZipFile(cand_zip, "r") as z:
            z.extractall(out)
        return out
    return None

def normalize_label_value(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    # remove punctuation and multiple spaces
    s = re.sub(r'[^a-z0-9]', '', s)
    return s

def find_label_column(df):
    lower_cols = {c.lower(): c for c in df.columns}
    for key in ("label","attack","class","category"):
        if key in lower_cols:
            return lower_cols[key]
    return df.columns[-1]

def main():
    raw_dir = find_raw_dir()
    if raw_dir is None:
        raise FileNotFoundError(
            f"Raw CICIoT folder not found. Put CSVs into one of: {POSSIBLE_RAW_DIRS} or place a CICIoT2023.zip in data/"
        )
    print("Using raw dir:", raw_dir)

    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"Found {len(csv_files)} CSV files. Loading (may take a while)...")
    dfs = []
    for p in csv_files:
        print(" ->", os.path.basename(p))
        try:
            dfs.append(pd.read_csv(p, low_memory=False))
        except Exception as e:
            print("   [WARN] Skipping", p, "due to:", e)

    if not dfs:
        raise RuntimeError("No CSVs loaded successfully.")

    df = pd.concat(dfs, ignore_index=True)
    print("Combined shape:", df.shape)

    label_col = find_label_column(df)
    print("Detected label column:", label_col)

    # normalize label strings
    y_raw = df[label_col].astype(str)
    y_norm = y_raw.map(normalize_label_value)

    # tokens recognized as benign after normalization
    benign_tokens = set([
        "benign", "benigntraffic", "benign_traffic", "normal", "normaltraffic", "normal_traffic",
        "ben", "benignflow", "benignflow"  # add more if you see other variants
    ])

    # mark benign where normalized label matches any benign token
    is_benign = y_norm.isin(benign_tokens)

    print("Detected benign samples (approx):", int(is_benign.sum()), "out of", len(df))

    if is_benign.sum() == 0:
        # show sample unique tokens to help debugging
        uniq = sorted(set(y_norm.tolist()))
        print("No benign tokens found. Sample normalized label tokens (first 50):")
        print(uniq[:50])
        raise RuntimeError("No benign rows detected. Please ensure the raw dataset contains BENIGN.csv or similar.")

    # make binary label: 0 = benign, 1 = attack
    y_bin = (~is_benign).astype(int)

    # keep only numeric feature columns
    X_num = df.select_dtypes(include=["number"]).copy()
    if X_num.shape[1] == 0:
        raise RuntimeError("No numeric feature columns detected. Check raw CSV format.")

    X_num["label"] = y_bin

    # If MAX_ROWS_PER_CLASS is set, build balanced sample for speed
    if MAX_ROWS_PER_CLASS is not None:
        benign_df = X_num[X_num["label"] == 0]
        attack_df = X_num[X_num["label"] == 1]
        n_b = len(benign_df)
        n_a = len(attack_df)
        print(f"Benign count: {n_b}, Attack count: {n_a}")
        nb = min(n_b, MAX_ROWS_PER_CLASS)
        na = min(n_a, MAX_ROWS_PER_CLASS)
        print(f"Sampling {nb} benign and {na} attack rows for balanced dataset...")
        benign_s = benign_df.sample(n=nb, random_state=42)
        attack_s = attack_df.sample(n=na, random_state=42)
        out_df = pd.concat([benign_s, attack_s], ignore_index=True).sample(frac=1, random_state=42)
    else:
        out_df = X_num

    print("Final dataset shape:", out_df.shape)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print("Saved processed & balanced dataset to:", OUT_PATH)

if __name__ == "__main__":
    main()
