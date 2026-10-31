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