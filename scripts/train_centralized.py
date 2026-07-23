import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import DATA_PATH, TEST_SIZE, RANDOM_STATE, RESULTS_DIR
from models.hybrid_model import build_model

def load_dataset():
    df = pd.read_csv(DATA_PATH)
    # assume last column is label
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # For CNN+LSTM, make (batch, seq_len, features_per_step)
    num_features_total = X.shape[1]
    seq_len = 10
    feat_per_step = int(np.ceil(num_features_total / seq_len))
    padded = np.zeros((X.shape[0], seq_len * feat_per_step), dtype=np.float32)
    padded[:, :num_features_total] = X
    X_seq = padded.reshape(X.shape[0], seq_len, feat_per_step)

    num_classes = len(np.unique(y))
    return X_seq, y, num_classes, feat_per_step

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X, y, num_classes, feat_per_step = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    model = build_model(num_features=feat_per_step, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 5

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{EPOCHS} - Train loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.cpu().numpy())

    report = classification_report(all_true, all_preds, digits=4)
    print("\nCentralized model classification report:\n")
    print(report)

    # Save to file
    with open(os.path.join(RESULTS_DIR, "centralized_report.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()
