import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

detections = np.load("detections.npy").astype(np.float32)
labels = np.load("observables.npy")[:, 0].astype(np.float32)
input_size = detections.shape[1]

print(f"Detections: {detections.shape}")
print(f"Logical error fraction: {labels.mean():.4f}")

X_train, X_test, y_train, y_test = train_test_split(
    detections, labels, test_size=0.2, random_state=42, stratify=labels
)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).unsqueeze(1)
y_test = torch.FloatTensor(y_test).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

class BinaryDecoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = BinaryDecoder(input_size=input_size).to(device)
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

model.eval()
with torch.no_grad():
    probs = model(X_test.to(device)).cpu()
    preds = (probs > 0.5).float()
    truth = y_test

    accuracy = (preds == truth).float().mean().item()
    baseline = 1 - truth.mean().item()

    tp = ((preds == 1) & (truth == 1)).sum().item()
    fp = ((preds == 1) & (truth == 0)).sum().item()
    fn = ((preds == 0) & (truth == 1)).sum().item()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

print(f"\nTest accuracy: {accuracy * 100:.2f}%")
print(f"Always-no-error baseline: {baseline * 100:.2f}%")
print(f"Recall on logical errors: {recall * 100:.2f}%")
print(f"Precision on logical errors: {precision * 100:.2f}%")

torch.save(model.state_dict(), "decoder_model.pth")
print("Model saved!")