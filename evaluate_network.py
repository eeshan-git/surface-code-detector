import stim
import numpy as np
import torch
import torch.nn as nn

distance = 5
rounds = 5
p = 0.001

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=rounds,
    distance=distance,
    after_clifford_depolarization=p,
    before_round_data_depolarization=p,
    before_measure_flip_probability=p,
    after_reset_flip_probability=p,
)

sampler = circuit.compile_detector_sampler(seed=12345)
detections, observables = sampler.sample(shots=200000, separate_observables=True)
truth = observables[:, 0]

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

model = BinaryDecoder(input_size=detections.shape[1])
model.load_state_dict(torch.load("decoder_model.pth", map_location="cpu"))
model.eval()

X = torch.FloatTensor(detections.astype(np.float32))
with torch.no_grad():
    probs = model(X).numpy()[:, 0]

pred = (probs > 0.5).astype(np.uint8)
net_error_rate = np.mean(pred != truth)

print(f"Shots: {len(truth)}")
print(f"Network logical error rate: {net_error_rate * 100:.3f}%")
print(f"Network failures: {int(np.sum(pred != truth))}")