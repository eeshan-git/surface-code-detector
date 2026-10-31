# Surface Code Decoder: PyMatching vs Neural Decoders

## Question
Can a neural network decode errors on a rotated surface code as well as PyMatching, a standard minimum weight matching decoder?

## Setup
Rotated surface code memory experiment from Stim (surface_code:rotated_memory_z), 5 rounds. Circuit level noise with physical error rate p = 0.001, including depolarization after Clifford gates, data qubit depolarization before each round, and measurement and reset flip errors. The network inputs are detection events and the label is whether the logical observable flipped.

Training: 200,000 shots per distance (50,000 for the distance 3 flat network), unseeded sampler, 80/20 split.
Evaluation: 200,000 fresh shots with seed 12345. All decoders are scored on the same shots.

## Results (failures out of 200,000 shots)

| Distance | PyMatching | Flat network | Convolutional network |
|---|---|---|---|
| 3 | 238 (0.12%) | 745 (0.37%) | 544 (0.27%) |
| 5 | 25 (0.01%) | 1,057 (0.53%) | 735 (0.37%) |

## Limitations
Only one noise rate and two code distances were tested, so this shows a trend, not a threshold. The learned decoders were trained on modest data. PyMatching's failure counts are small, so its rates carry statistical noise.

## How to reproduce

1. `python generate_detectors.py` (generates detection events and logical labels)
2. `python train_decoder_detectors.py` (trains the flat network)
3. `python train_conv_decoder.py` (trains the convolutional network)
4. `python evaluate_pymatching.py` and `python evaluate_network.py` / `python evaluate_conv.py` (evaluates on the seeded shots)

Install with `pip install stim pymatching torch scikit-learn numpy matplotlib`.

### Before you run

Each script sets its distance at the top with `distance = 3` or `DISTANCE = 3`. The four main scripts were run at both distances, so change the value to match the run you want. Keep `p = 0.001` and `rounds = 5` as they are, since all results were produced with those settings.

The training and evaluation scripts read files by name, so the file names have to match:

1. `generate_detectors.py` writes `detections.npy` and `observables.npy`. Rename these to `detections_d{distance}_200k.npy` and `observables_d{distance}_200k.npy` after generating, for example `detections_d5_200k.npy` for distance 5.
2. `train_decoder_detectors.py` reads `detections.npy` and `observables.npy`. Run it before renaming, or change its file names to match your renamed files.
3. `train_conv_decoder.py` and `evaluate_conv.py` read the distance-named files and the saved model `decoder_conv_d{distance}.pth`.
4. `evaluate_network.py` loads `decoder_model.pth`, so it is set to distance 5. It must be run after `train_decoder_detectors.py` at distance 5.

Generated data (`.npy`) and model files (`.pth`) are not included in the repo. Each step regenerates them, and they can take several minutes to produce.
