import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from mouse_transformer_model import MouseTransformer


class MousePathDataset(Dataset):
    def __init__(self, json_path, sequence_length=20):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.sequences = []
        self.targets = []
        self.seq_len = sequence_length

        for trial in data:
            path = trial["normalized_path"]
            path = np.array(path)

            for i in range(len(path) - sequence_length):
                input_seq = path[i:i + sequence_length]
                target = path[i + sequence_length]
                self.sequences.append(input_seq)
                self.targets.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# --- Training Function ---
def train_mouse_transformer(dataset, model, batch_size=32, num_epochs=20, learning_rate=1e-4, device='cpu'):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = []
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

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'avg_dt': np.mean([step[0] for trial in dataset.sequences for step in trial])
    }, "models/mouse_transformer.pth")

    return model, loss_history

# --- Run Training ---
data_path = "mouse_data_normalized/normalized_mouse_data_1743325848.json"
dataset = MousePathDataset(data_path, sequence_length=20)
model = MouseTransformer()
trained_model, loss_history = train_mouse_transformer(
    dataset=dataset,
    model=model,
    batch_size=32,
    num_epochs=30,
    learning_rate=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


