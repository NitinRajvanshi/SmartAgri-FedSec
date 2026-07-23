# 🛡️ SmartAgri-FedSec
### Privacy-Preserving Intrusion Detection System for Smart Agriculture using Hybrid Deep Learning, Federated Learning, and Homomorphic Encryption

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-success)
![Homomorphic Encryption](https://img.shields.io/badge/Homomorphic-Paillier-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Project Overview

SmartAgri-FedSec is a privacy-preserving Intrusion Detection System (IDS) designed for Smart Agriculture Internet of Things (IoT) environments.

The project combines modern Artificial Intelligence and Privacy-Preserving Machine Learning techniques to detect malicious network traffic while ensuring that sensitive agricultural data never leaves local devices.

Unlike traditional centralized machine learning approaches, SmartAgri-FedSec uses **Federated Learning (FL)** to train models across multiple simulated agricultural clients without sharing raw data. To further strengthen privacy, the project integrates **Paillier Homomorphic Encryption (HE)**, enabling encrypted model aggregation during federated learning.

The intrusion detection model is based on a **Hybrid CNN–LSTM–DNN architecture**, allowing it to capture spatial patterns, sequential dependencies, and complex decision boundaries in IoT network traffic.

---

## 🎯 Problem Statement

Modern smart agriculture relies heavily on IoT devices such as:

- 🌱 Soil moisture sensors
- 🌡 Temperature sensors
- 💧 Smart irrigation systems
- 📷 Smart surveillance cameras
- 🚜 Autonomous farming equipment

These connected devices continuously exchange data across the network.

Unfortunately, they are vulnerable to cyber attacks such as:

- Distributed Denial of Service (DDoS)
- Malware Infections
- Botnet Attacks
- Unauthorized Access
- Network Intrusions

Traditional intrusion detection systems require collecting all data on a centralized server, creating significant privacy risks and communication overhead.

SmartAgri-FedSec addresses this challenge by enabling collaborative model training without exposing sensitive agricultural data.

---

## 💡 Proposed Solution

This project integrates three modern technologies into a single intelligent security framework:

- 🧠 Hybrid CNN–LSTM–DNN Deep Learning Model
- 🌐 Federated Learning (FedAvg)
- 🔒 Paillier Homomorphic Encryption

The overall workflow enables multiple agricultural clients to collaboratively train a global intrusion detection model while preserving data privacy throughout the training process.
