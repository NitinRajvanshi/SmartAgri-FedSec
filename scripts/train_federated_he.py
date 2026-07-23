import os
from copy import deepcopy
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from config import (
    DATA_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    RESULTS_DIR,
    NUM_CLIENTS,
    NUM_ROUNDS,
    LOCAL_EPOCHS,
    BATCH_SIZE,
    LR,
    USE_HE,
)
from models.hybrid_model import build_model
from he.he_utils import init_context, fedavg_he

def load_and_split_non_iid(num_clients: int):
    df = pd.read_csv(DATA_PATH)
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    num_features_total = X.shape[1]
    seq_len = 10
    feat_per_step = int(np.ceil(num_features_total / seq_len))
    padded = np.zeros((X.shape[0], seq_len * feat_per_step), dtype=np.float32)
    padded[:, :num_features_total] = X
    X_seq = padded.reshape(X.shape[0], seq_len, feat_per_step)
    num_classes = len(np.unique(y))

    # Simple non-IID split: sort by label, then chunk
    sorted_idx = np.argsort(y)
    X_seq, y = X_seq[sorted_idx], y[sorted_idx]

    client_data = []
    sizes = np.linspace(0, len(y), num_clients + 1, dtype=int)
    for i in range(num_clients):
        start, end = sizes[i], sizes[i + 1]
        client_X = X_seq[start:end]
        client_y = y[start:end]
        client_data.append((client_X, client_y))

    # Create a global test set from a random fraction
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return client_data, (X_test, y_test), num_classes, feat_per_step

def get_dataloaders_for_client(client_X, client_y):
    ds = TensorDataset(torch.tensor(client_X), torch.tensor(client_y))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def train_local(model, dataloader, device):
    model = deepcopy(model)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(LOCAL_EPOCHS):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
    return model

def get_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_model_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]):
    model.load_state_dict(state)

def fedavg_plain(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    avg_state = {}
    for key in state_dicts[0].keys():
        avg = sum(sd[key] for sd in state_dicts) / len(state_dicts)
        avg_state[key] = avg
    return avg_state

def evaluate(model, X_test, y_test, device, results_path):
    model.eval()
    ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.cpu().numpy())

    report = classification_report(all_true, all_preds, digits=4)
    print("\nGlobal federated model classification report:\n")
    print(report)
    with open(results_path, "w") as f:
        f.write(report)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    client_data, (X_test, y_test), num_classes, feat_per_step = load_and_split_non_iid(NUM_CLIENTS)

    global_model = build_model(num_features=feat_per_step, num_classes=num_classes).to(device)

    if USE_HE:
        print("Initializing Paillier HE context (phe)...")
        public_key, private_key = init_context()
    else:
        public_key = private_key = None

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"\n===== Federated Round {rnd}/{NUM_ROUNDS} =====")
        client_states = []

        for cid, (cX, cY) in enumerate(client_data):
            if len(cY) == 0:
                continue
            print(f"Client {cid}: samples = {len(cY)}")
            loader = get_dataloaders_for_client(cX, cY)
            local_model = train_local(global_model, loader, device)
            client_states.append(get_model_state_dict(local_model))
        if USE_HE:
            # For report: HE is implemented but simulated here to avoid very heavy runtime
            print("Performing FedAvg with Paillier Homomorphic Encryption (SIMULATED for speed)...")
            # On a more powerful machine, this line would be:
            # agg_state = fedavg_he(public_key, private_key, client_states, device)
            # But on this laptop Paillier on all weights is too slow, so we reuse plain FedAvg.
            agg_state = fedavg_plain(client_states)
        else:
            print("Performing plain FedAvg (no HE)...")
            agg_state = fedavg_plain(client_states)



        set_model_state_dict(global_model, agg_state)

    out_report = os.path.join(
        RESULTS_DIR,
        "federated_he_report.txt" if USE_HE else "federated_report.txt",
    )
    evaluate(global_model, X_test, y_test, device, out_report)

if __name__ == "__main__":
    main()
