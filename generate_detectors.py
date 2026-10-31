import stim
import numpy as np

distance = 3
rounds = 5
p = 0.001 # physical error rate

circuit = stim.Circuit.generated(
    "surface_code:rotated_memory_z",
    rounds=rounds,
    distance=distance,
    after_clifford_depolarization=p,
    before_round_data_depolarization=p,
    before_measure_flip_probability=p,
    after_reset_flip_probability=p,
)

sampler = circuit.compile_detector_sampler()
detections, observables = sampler.sample(shots=200000, separate_observables=True)

np.save("detections.npy", detections.astype(np.uint8))
np.save("observables.npy", observables.astype(np.uint8))

print(f"Detections shape: {detections.shape}")
print(f"Observables shape: {observables.shape}")
print(f"Fraction with logical error: {observables.mean():.4f}")