import torch
import torch.nn as nn

class HybridCNNLSTM(nn.Module):
    """Hybrid 1D-CNN + LSTM + DNN model for intrusion detection.

    Input expected shape: (batch, seq_len, num_features)
    We treat num_features as channels in 1D-CNN by transposing.
    """

    def __init__(self, num_features: int, num_classes: int):
        super().__init__()

        # CNN block
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM block
        self.lstm_hidden = 64
        self.lstm = nn.LSTM(
            input_size=32,  # matches CNN out_channels
            hidden_size=self.lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # DNN head
        self.fc1 = nn.Linear(self.lstm_hidden, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        # CNN expects (batch, channels, seq_len)
        x = x.transpose(1, 2)  # -> (batch, num_features, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)  # (batch, 32, seq_len//2)

        # LSTM expects (batch, seq_len, features)
        x = x.transpose(1, 2)  # -> (batch, seq_len//2, 32)

        lstm_out, _ = self.lstm(x)  # (batch, seq_len//2, hidden)
        # Take last time step
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)

        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def build_model(num_features: int, num_classes: int) -> nn.Module:
    return HybridCNNLSTM(num_features=num_features, num_classes=num_classes)
