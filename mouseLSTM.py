import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os

data_path = "mouse_data_normalized/normalized_mouse_data_1743325848.json"


class MousePathDataset(Dataset):
    def __init__(self, json_path, sequence_length=20):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.sequences = []
        self.targets = []
        self.seq_len = sequence_length

        self.all_dt = []  # For avg_dt calculation

        for trial in data:
            path = trial["normalized_path"]
            path = np.array(path)

            # Store dt values for later avg_dt computation
            self.all_dt.extend(path[:, 0])  # ∆t is first column

            # Create sliding windows
            for i in range(len(path) - sequence_length):
                input_seq = path[i:i + sequence_length]
                target = path[i + sequence_length]
                self.sequences.append(input_seq)
                self.targets.append(target)

        self.avg_dt = float(np.mean(self.all_dt))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),  # shape: [seq_len, 3]
            torch.tensor(self.targets[idx], dtype=torch.float32)  # shape: [3]
        )


class MouseLSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=2, dropout=0.2):
        super(MouseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 3)  # Predict next ∆t, ∆x, ∆y

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)


def train_mouse_model(dataset, model, batch_size=32, num_epochs=10, learning_rate=1e-3, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

    os.makedirs("models", exist_ok=True)

    # Save model and avg_dt in a checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "avg_dt": dataset.avg_dt
    }, "models/mouse_lstm_checkpoint.pth")

    print(f"Model and avg_dt saved to models/mouse_lstm_checkpoint.pth")
    return model


# Load dataset and train
dataset = MousePathDataset(data_path, sequence_length=20)
model = MouseLSTM()
trained_model = train_mouse_model(
    dataset=dataset,
    model=model,
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
