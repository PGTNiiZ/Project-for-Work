# R0/R1/R2/R3 Experiment Report

R1 and R3 deployment genomes were selected from GA-validation cost only; test metrics were not used for seed/genome selection.

| Experiment | Score/model | F1 (mean +/- sd) | Cost (mean +/- sd) |
|---|---|---:|---:|
| R0 | Original 17-feature score / production manual rule | 0.7964 +/- 0.0000 | 2304.66 +/- 0.00 |
| R1 | Original 17-feature score / GA rule | 0.8563 +/- 0.0001 | 2329.76 +/- 0.31 |
| R2 | Frozen 18-feature MLP + MiniLM cosine / manual 0.98/0.95 | 0.8224 +/- 0.0000 | 2377.24 +/- 0.00 |
| R3 | Same frozen R2 probabilities / GA rule | 0.8615 +/- 0.0001 | 1917.39 +/- 0.01 |

## Selected validation-only genomes

- R1: seed 999 (validation cost 541.42)
- R3: seed 7 (validation cost 559.02)

## Figures

- r2_training_history.png
- r1_r3_ga_history.png

## Model-seed robustness

Within-model GA-seed standard deviations are recorded per model below; the final line is the sample standard deviation across model-seed means.

| Model seed | F1 mean over GA seeds | F1 SD within GA seeds | Cost mean over GA seeds | Cost SD within GA seeds |
|---:|---:|---:|---:|---:|
| 7 | 0.8642 | 0.0041 | 2027.66 | 37.89 |
| 42 | 0.8615 | 0.0001 | 1917.39 | 0.01 |
| 123 | 0.8620 | 0.0001 | 2003.92 | 14.66 |

Across model-seed means: F1 SD 0.0015; cost SD 58.04.
