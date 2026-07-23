# SmartAgri-FedSec (Alt HE Version): CNN–LSTM–DNN + Federated Learning + Homomorphic Encryption (Paillier)

This version of the project removes the heavy **Pyfhel/CKKS** dependency and instead uses
a lightweight **Paillier Homomorphic Encryption** library (`phe`) so that it works on
Windows + normal Python (no C++ toolchain needed).

You still get **full integration of**:

- Hybrid **CNN–LSTM–DNN** intrusion detection model
- **Federated Learning** (simulated clients + FedAvg)
- **Homomorphic Encryption (HE)** using Paillier (additively homomorphic)

---

## 1. Install Dependencies (no conda required)

Open CMD in your project folder (where this README is):

```bash
cd path\to\smartagri_fedsec_project_alt

python -m pip install --upgrade pip

pip install -r requirements.txt
```

`requirements.txt` only contains pure-Python or wheel-based packages plus **phe** (Paillier HE),
so it should install without any C++ build tools.

> If you have multiple Python versions, you may need to use `py -m pip` instead of `python -m pip`.

---

## 2. Dataset

A small synthetic IoT intrusion dataset is already provided at:

```text
data/iot_data.csv
```

- 1000 samples
- 20 numeric features `f0`..`f19`
- Binary label column `label` (0 = normal, 1 = attack)

You can later replace it with a known dataset (e.g. N-BaIoT, CICIoT) if you want, as long as:
- All feature columns are numeric
- The last column is the label

---

## 3. Run Centralized Training (CNN–LSTM–DNN)

```bash
python train_centralized.py
```

This trains the hybrid model on the entire dataset (no FL, no HE) and prints a classification report.
The report is also saved into:

```text
results/centralized_report.txt
```

---

## 4. Run Federated Learning + Homomorphic Encryption

The configuration is in `config.py`. By default:

```python
NUM_CLIENTS = 5
NUM_ROUNDS = 3
LOCAL_EPOCHS = 1
USE_HE = True
```

To run the federated + HE experiment:

```bash
python train_federated_he.py
```

What happens:

1. The global CNN–LSTM–DNN model is initialized.
2. Dataset is split into `NUM_CLIENTS` non-IID parts (simulating different farms/gateways).
3. Each client trains the model locally for `LOCAL_EPOCHS` epochs.
4. Clients send **encrypted** model parameters (weights) to the server using Paillier HE.
5. The server **homomorphically adds** encrypted parameters element-wise and then decrypts
   the sum to obtain the averaged global model (FedAvg + HE).
6. The global model is evaluated on a held-out test set.

The final classification report is saved to:

```text
results/federated_he_report.txt
```

---

## 5. Where HE is implemented (for your report/viva)

- **Model (CNN–LSTM–DNN)**: `models/hybrid_model.py`
- **HE utilities (Paillier)**: `he/he_utils.py`
- **Federated training + HE integration**: `train_federated_he.py`

In short:

- Each client's model parameters are quantized (scaled floats → integers).
- Integers are encrypted using Paillier (`phe`).
- The server performs element-wise addition **directly on ciphertexts**.
- After aggregation, the server decrypts, rescales back to floats, and applies FedAvg.

This demonstrates a complete pipeline of **Hybrid Deep IDS + Federated Learning + Homomorphic Encryption**
matching your project description, while being runnable on normal Windows Python.
