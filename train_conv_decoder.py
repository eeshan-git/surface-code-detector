import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from grid_utils import build_circuit, detector_grid_index, to_grid

DISTANCE = 3   # set to 5 to train the distance 5 model
ROUNDS = 5
P = 0.001

circuit = build_circuit(distance=DISTANCE, rounds=ROUNDS, p=P)
idx, shape = detector_grid_index(circuit)
print(f"Distance {DISTANCE}, grid (time, x, y): {shape}")

detections = np.load(f"detections_d{DISTANCE}_200k.npy")
labels = np.load(f"observables_d{DISTANCE}_200k.npy")[:, 0].astype(np.float32)
grid = to_grid(detections, idx, shape)
print(f"Grid tensor: {grid.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    grid, labels, test_size=0.2, random_state=42, stratify=labels
)
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

class ConvDecoder(nn.Module):
    def __init__(self, time_steps, gx, gy):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(time_steps, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * gx * gy, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.conv(x))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = ConvDecoder(shape[0], shape[1], shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

epochs = 30
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), f"decoder_conv_d{DISTANCE}.pth")
print(f"Model saved to decoder_conv_d{DISTANCE}.pth")