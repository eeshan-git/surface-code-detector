# Surface Code Decoder: PyMatching vs Neural Decoders

## Question
Can a neural network decode errors on a rotated surface code as well as PyMatching, a standard minimum weight matching decoder?

## Setup
Rotated surface code memory experiment from Stim (surface_code:rotated_memory_z), 5 rounds. Circuit level noise with physical error rate p = 0.001, including depolarization after Clifford gates, data qubit depolarization before each round, and measurement and reset flip errors. The network inputs are detection events and the label is whether the logical observable flipped.

Training: 200,000 shots per network, drawn with an unseeded sampler, so training shots differ from the evaluation shots. 80/20 split.
Evaluation: 200,000 fresh shots with seed 12345. All decoders are scored on the same shots.

## Results (failures out of 200,000 shots)

| Distance | PyMatching | Flat network | Convolutional network |
|---|---|---|---|
| 3 | 238 (0.12%) | 413 (0.21%) | 544 (0.27%) |
| 5 | 25 (0.01%) | 1,057 (0.53%) | 735 (0.37%) |

## Limitations
Only one noise rate and two code distances were tested, so this shows a trend, not a threshold. The learned decoders were trained on modest data. PyMatching's failure counts are small, so its rates carry statistical noise. At distance 3 the flat network beat the convolutional network once both were trained on 200,000 shots, so the convolutional advantage holds only at distance 5.

## Files
- `generate_detectors.py`: generates detection events and logical labels
- `train_decoder_detectors.py`: trains the flat network
- `train_conv_decoder.py`: trains the convolutional network (needs `grid_utils.py`)
- `evaluate_pymatching.py`, `evaluate_network.py`, `evaluate_conv.py`: evaluate on the seeded shots
- `grid_utils.py`: shared circuit builder and detector-to-grid helper, required by the convolutional scripts

## How to reproduce
Install with `pip install stim pymatching torch scikit-learn numpy matplotlib`. Keep all files in the same folder. The steps below reproduce the distance 5 results.

1. `python generate_detectors.py` at distance 5.
2. `python train_decoder_detectors.py` trains the flat network. It reads `detections.npy`, so run it now, before renaming.
3. Rename `detections.npy` to `detections_d5_200k.npy` and `observables.npy` to `observables_d5_200k.npy`.
4. `python train_conv_decoder.py` trains the convolutional network. It reads the distance-named files from step 3.
5. `python evaluate_pymatching.py`, `python evaluate_network.py`, and `python evaluate_conv.py` evaluate on the seeded shots.

For distance 3, repeat the same steps with `distance = 3` (or `DISTANCE = 3`) in each script, and use `_d3_200k` in the file names. Train the flat network on the 200,000-shot distance 3 data, and save its model as `decoder_model_d3_200k.pth`. In `evaluate_network.py`, set `distance = 3` and load `decoder_model_d3_200k.pth`.

### Before you run

Keep `p = 0.001` and `rounds = 5` as they are. Each script sets its distance at the top (`distance = 3` or `DISTANCE = 5`).

Data (`.npy`) and model (`.pth`) files are not in the repo. Each step regenerates them.
