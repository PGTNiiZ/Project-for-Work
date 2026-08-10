# GB-17 / GB-18 / GB-18-GA Experiment

Nested entity-aware split seed 42; outer test remained sealed until final evaluation.
GB configuration: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42).

## Test comparison

| Experiment | TP | FP | FN | REVIEW | Precision | Recall | F1 | Weighted cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 5864 | 88 | 1689 | 8783 | 0.9852 | 0.6683 | 0.7964 | 2304.6600 |
| R3 | 6689.6000 | 66.0000 | 1565.0000 | 1119.4000 | 0.9902 | 0.7624 | 0.8615 | 1917.3880 |
| GB-17 | 3736 | 11 | 1053 | 0 | 0.9971 | 0.4258 | 0.5967 | 1108.0000 |
| GB-18 | 3736 | 11 | 1053 | 0 | 0.9971 | 0.4258 | 0.5967 | 1108.0000 |
| GB-18-GA | 3769 | 12 | 1008 | 41 | 0.9968 | 0.4295 | 0.6004 | 1068.8200 |

Winner by weighted cost: **GB-18-GA**.

## Isolated effects (right minus left; cost reduction is positive when cost falls)

### GB-17_to_GB-18_MiniLM_effect

TP=0.0000, FP=0.0000, FN=0.0000, REVIEW=0.0000, precision=0.0000, recall=0.0000, F1=0.0000, cost=0.0000, cost_reduction=0.0000

### GB-18_to_GB-18-GA_decision_layer_effect

TP=33.0000, FP=1.0000, FN=-45.0000, REVIEW=41.0000, precision=-0.0003, recall=0.0037, F1=0.0037, cost=-39.1800, cost_reduction=39.1800

### R3_to_GB-18-GA_model_family_effect

TP=-2920.6000, FP=-54.0000, FN=-557.0000, REVIEW=-1078.4000, precision=0.0066, recall=-0.3329, F1=-0.2611, cost=-848.5680, cost_reduction=848.5680

## GA validation seeds

Test metrics were not calculated per GA seed. The selected seed/genome was chosen from ga_validation weighted cost only, then evaluated on test once.

| Seed | Validation cost | t_m | t_r | c_promote | c_demote |
|---:|---:|---:|---:|---:|---:|
| 7 | 404.06 | 0.936584 | 0.501494 | 0.920210 | 0.095064 |
| 42 | 379.24 | 0.999000 | 0.641810 | 0.960860 | 0.508509 |
| 123 | 379.24 | 0.981600 | 0.585505 | 0.961672 | 0.408355 |
| 999 | 485.00 | 0.916940 | 0.860770 | 0.570891 | 0.366737 |
| 2025 | 379.24 | 0.977807 | 0.561375 | 0.966059 | 0.532750 |

Selected seed: 42; selected validation cost: 379.24.
Selected genome: `{"c_demote": 0.5085094888084053, "c_promote": 0.9608596315630417, "t_m": 0.999, "t_r": 0.6418103872519474}`.

## Verification

- GB-17 and GB-18 share the same sampled training pair hash, labels, model seed, and hyperparameters.
- GB-18 manual and GB-18-GA use the same `gb18_probabilities.parquet` artifact.
- Test labels were not accessed before final selection; drop pairs were excluded.
- See `artifact_hashes.json` for SHA-256 hashes.
