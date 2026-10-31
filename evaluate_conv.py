import numpy as np
import torch
import torch.nn as nn
from grid_utils import build_circuit, detector_grid_index, to_grid

DISTANCE = 5   # must match the training run
ROUNDS = 5
P = 0.001

circuit = build_circuit(distance=DISTANCE, rounds=ROUNDS, p=P)
idx, shape = detector_grid_index(circuit)

sampler = circuit.compile_detector_sampler(seed=12345)
detections, observables = sampler.sample(shots=200000, separate_observables=True)
truth = observables[:, 0]
grid = to_grid(detections.astype(np.float32), idx, shape)

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

model = ConvDecoder(shape[0], shape[1], shape[2])
model.load_state_dict(torch.load(f"decoder_conv_d{DISTANCE}.pth", map_location="cpu"))
model.eval()

with torch.no_grad():
    probs = model(torch.FloatTensor(grid)).numpy()[:, 0]

pred = (probs > 0.5).astype(np.uint8)
failures = int(np.sum(pred != truth))

print(f"Distance: {DISTANCE}")
print(f"Shots: {len(truth)}")
print(f"Conv network logical error rate: {failures / len(truth) * 100:.3f}%")
print(f"Conv network failures: {failures}")
