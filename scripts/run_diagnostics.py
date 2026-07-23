# run_diagnostics.py
import pandas as pd
from collections import Counter
import os

BASE = os.path.dirname(os.path.abspath(__file__))
csv = os.path.join(BASE, "data", "iot_data_ciciot2023.csv")   # or your iot_data.csv path
df = pd.read_csv(csv, low_memory=False)
label_col = df.columns[-1]
print("Using file:", csv)
print("Label column:", label_col)
counts = Counter(df[label_col].astype(str).values)
print("Label value counts:", counts)
# show numeric range
try:
    ys = df[label_col].astype(int)
    print("min label, max label, unique:", ys.min(), ys.max(), sorted(ys.unique()))
except Exception as e:
    print("Could not interpret labels as ints:", e)
