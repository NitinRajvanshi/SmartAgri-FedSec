# 🛡️ FedSecure-IoT IDS

### Privacy-Preserving Intrusion Detection System for IoT Networks using Hybrid Deep Learning, Federated Learning, and Homomorphic Encryption

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-success)
![Homomorphic Encryption](https://img.shields.io/badge/Paillier-Encryption-orange)
![IoT Security](https://img.shields.io/badge/IoT-Cybersecurity-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

---

# 📖 Overview

FedSecure-IoT IDS is a privacy-preserving Intrusion Detection System (IDS) designed for modern Internet of Things (IoT) environments.

The project combines Deep Learning, Federated Learning, and Homomorphic Encryption to detect malicious network traffic while ensuring sensitive device data remains private.

Instead of collecting data from every IoT device on a central server, the system enables distributed model training across multiple clients using Federated Learning. During model aggregation, Paillier Homomorphic Encryption protects model updates, allowing secure collaborative learning without exposing private information.

The proposed framework is suitable for various IoT ecosystems including:

- Smart Homes
- Smart Cities
- Industrial IoT (IIoT)
- Healthcare IoT
- Connected Vehicles
- Smart Agriculture
- Critical Infrastructure



# 🎯 Problem Statement

Modern IoT ecosystems consist of thousands of interconnected devices continuously exchanging data across distributed networks.

These environments are increasingly targeted by cyber threats including:

- Distributed Denial of Service (DDoS)
- Botnet Attacks
- Malware
- Unauthorized Access
- Network Intrusions
- Device Compromise

Traditional machine learning solutions require collecting all training data on a centralized server. While effective, this approach introduces several challenges:

- Privacy concerns
- Regulatory compliance issues
- High communication costs
- Single point of failure
- Risk of sensitive data exposure

There is a growing need for intelligent intrusion detection systems that preserve privacy while maintaining high detection performance.



# 💡 Proposed Solution

FedSecure-IoT IDS addresses these challenges by integrating three advanced technologies:

- 🧠 Hybrid CNN–LSTM–DNN for intrusion detection
- 🌐 Federated Learning (FedAvg) for decentralized model training
- 🔒 Paillier Homomorphic Encryption for secure model aggregation

Each IoT client trains its local model using private network traffic.

Only encrypted model parameters are shared with the federated server.

The server aggregates encrypted parameters without accessing raw client data, producing an improved global intrusion detection model while preserving user privacy.
The intrusion detection engine is based on a Hybrid CNN–LSTM–DNN architecture capable of learning complex spatial and temporal patterns from network traffic to accurately classify normal and malicious behavior.

# ✨ Features

- Hybrid CNN–LSTM–DNN based intrusion detection model
- Federated Learning using the Federated Averaging (FedAvg) algorithm
- Privacy-preserving model aggregation using Paillier Homomorphic Encryption
- Supports both centralized and federated training workflows
- Configurable number of clients, communication rounds, and local epochs
- Performance evaluation using standard classification metrics
- Visualization of training performance and comparison results
- Modular Python implementation for easy experimentation and extension

  # 🛠 Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Machine Learning | Scikit-learn |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |
| Federated Learning | Custom FedAvg Implementation |
| Homomorphic Encryption | Paillier (phe) |
| Dataset | IoT Network Traffic Dataset |

# 📂 Project Structure

```
FedSecure-IoT/
│
├── data/              # Dataset
├── models/            # Hybrid CNN-LSTM-DNN model
├── scripts/           # Training and evaluation scripts
├── results/           # Reports and graphs
├── he/                # Homomorphic encryption utilities
├── config.py          # Project configuration
├── requirements.txt   # Dependencies
└── README.md
```
# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/NitinRajvanshi/FedSecure_IoT-IDS.git
cd FedSecure_IoT-IDS
```

## 2. Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

# ▶ Usage

## Run Centralized Training

```bash
python scripts/train_centralized.py
```

This trains the Hybrid CNN–LSTM–DNN model using centralized learning and generates a classification report.

---

## Run Federated Learning with Homomorphic Encryption

```bash
python scripts/train_federated_he.py
```

This simulates multiple federated clients, encrypts model parameters using Paillier Homomorphic Encryption, performs secure model aggregation, and evaluates the global model.

# ⚙ Configuration

Project settings can be modified in:

```text
config.py
```

Some configurable parameters include:

- Number of federated clients
- Number of communication rounds
- Local training epochs
- Learning rate
- Batch size
- Enable / Disable Homomorphic Encryption

  # 📊 Results

The project evaluates both centralized and federated learning approaches using standard machine learning metrics.

Generated outputs include:

- Classification Reports
- Accuracy Comparison
- F1-Score Comparison
- Training Metrics
- Performance Visualizations

All generated reports and figures are stored inside the `results/` directory.

# 👨‍💻 Author

**Nitin Kumar Rajvanshi**

- GitHub: https://github.com/NitinRajvanshi
- LinkedIn: https://www.linkedin.com/in/nitinkumarrajvanshi

  # 📜 License

This project is licensed under the MIT License.

See the `LICENSE` file for more details.


