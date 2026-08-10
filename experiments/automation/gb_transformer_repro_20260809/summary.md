# GB-17 / GB-18 / GB-18-GA Experiment

Nested entity-aware split seed 42; outer test remained sealed until final evaluation.
GB configuration: GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42).

## Test comparison

| Experiment | TP | FP | FN | REVIEW | Precision | Recall | F1 | Weighted cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0 | 5864 | 88 | 1689 | 8783 | 0.9852 | 0.6683 | 0.7964 | 2304.6600 |
| R3 | 6689.6000 | 66.0000 | 1565.0000 | 1119.4000 | 0.9902 | 0.7624 | 0.8615 | 1917.3880 |
| GB-17 | 3736 | 11 | 1916 | 3289 | 0.9971 | 0.4258 | 0.5967 | 2036.7800 |
| GB-18 | 6266 | 65 | 1833 | 755 | 0.9897 | 0.7141 | 0.8296 | 2173.1000 |
| GB-18-GA | 6711 | 99 | 1571 | 861 | 0.9855 | 0.7648 | 0.8612 | 2083.2200 |

Winner by weighted cost: **R3**.

## Isolated effects (right minus left; cost reduction is positive when cost falls)

### GB-17_to_GB-18_MiniLM_effect

TP=2530.0000, FP=54.0000, FN=-83.0000, REVIEW=-2534.0000, precision=-0.0074, recall=0.2883, F1=0.2329, cost=136.3200, cost_reduction=-136.3200

### GB-18_to_GB-18-GA_decision_layer_effect

TP=445.0000, FP=34.0000, FN=-262.0000, REVIEW=106.0000, precision=-0.0042, recall=0.0507, F1=0.0316, cost=-89.8800, cost_reduction=89.8800

### R3_to_GB-18-GA_model_family_effect

TP=21.4000, FP=33.0000, FN=6.0000, REVIEW=-258.4000, precision=-0.0047, recall=0.0024, F1=-0.0003, cost=165.8320, cost_reduction=-165.8320

## GA validation seeds

Test metrics were not calculated per GA seed. The selected seed/genome was chosen from ga_validation weighted cost only, then evaluated on test once.

| Seed | Validation cost | t_m | t_r | c_promote | c_demote |
|---:|---:|---:|---:|---:|---:|
| 7 | 612.84 | 0.974204 | 0.557505 | 0.956690 | 0.340904 |
| 42 | 611.88 | 0.973306 | 0.500000 | 0.956722 | 0.118752 |
| 123 | 611.88 | 0.974658 | 0.500000 | 0.956688 | 0.474875 |
| 999 | 611.84 | 0.971329 | 0.500000 | 0.956125 | 0.653156 |
| 2025 | 611.88 | 0.970811 | 0.500000 | 0.956810 | 0.156562 |

Selected seed: 999; selected validation cost: 611.84.
Selected genome: `{"c_demote": 0.6531558327331847, "c_promote": 0.9561245224549983, "t_m": 0.9713293771286314, "t_r": 0.5}`.

## Verification

- GB-17 and GB-18 share the same sampled training pair hash, labels, model seed, and hyperparameters.
- GB-18 manual and GB-18-GA use the same `gb18_probabilities.parquet` artifact.
- Test labels were not accessed before final selection; drop pairs were excluded.
- See `artifact_hashes.json` for SHA-256 hashes.
