import numpy as np
import pandas as pd

np.random.seed(42)

num_features = 20
n_normal = 800
n_attack = 400

# Normal traffic: smaller values
X_normal = np.random.normal(loc=0.0, scale=1.0, size=(n_normal, num_features))

# Attack traffic: larger values (shifted mean)
X_attack = np.random.normal(loc=2.5, scale=1.0, size=(n_attack, num_features))

X = np.vstack([X_normal, X_attack])
y = np.array([0]*n_normal + [1]*n_attack)

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(num_features)])
df["label"] = y

df.to_csv("data/iot_data.csv", index=False)
print("New synthetic dataset saved to data/iot_data.csv")
