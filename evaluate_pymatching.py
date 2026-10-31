import stim
import numpy as np
import pymatching

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

dem = circuit.detector_error_model(decompose_errors=True)
matcher = pymatching.Matching.from_detector_error_model(dem)

# Fixed seed, so this is a fresh sample rather than the training data
sampler = circuit.compile_detector_sampler(seed=12345)
detections, observables = sampler.sample(shots=200000, separate_observables=True)

predictions = matcher.decode_batch(detections)
truth = observables[:, 0]
pred = predictions[:, 0]

logical_error_rate = np.mean(pred != truth)

print(f"Shots: {len(truth)}")
print(f"Logical error fraction in data: {truth.mean() * 100:.2f}%")
print(f"PyMatching logical error rate: {logical_error_rate * 100:.2f}%")
print(f"PyMatching accuracy: {(1 - logical_error_rate) * 100:.2f}%")
print(f"PyMatching failures: {int(np.sum(predictions[:, 0] != truth))}")
