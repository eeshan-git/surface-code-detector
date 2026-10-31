import numpy as np
import stim

def build_circuit(distance=5, rounds=5, p=0.001):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        rounds=rounds,
        distance=distance,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )

def detector_grid_index(circuit):
    coords = circuit.get_detector_coordinates()
    n = circuit.num_detectors
    xyt = np.array([coords[i][:3] for i in range(n)])
    ts = np.unique(xyt[:, 2])
    xs = np.unique(xyt[:, 0])
    ys = np.unique(xyt[:, 1])
    idx = np.zeros((n, 3), dtype=int)
    idx[:, 0] = np.searchsorted(ts, xyt[:, 2])
    idx[:, 1] = np.searchsorted(xs, xyt[:, 0])
    idx[:, 2] = np.searchsorted(ys, xyt[:, 1])
    shape = (len(ts), len(xs), len(ys))
    return idx, shape

def to_grid(detections, idx, shape):
    grid = np.zeros((detections.shape[0],) + shape, dtype=np.float32)
    for i in range(detections.shape[1]):
        t, x, y = idx[i]
        grid[:, t, x, y] = detections[:, i]
    return grid