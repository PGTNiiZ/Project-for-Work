# GA Automation: r3_sensitivity_fp10_20260809

## ขั้นตอนการทดลอง

1. โหลด cache ที่แบ่ง entity-aware เป็น validation/test แล้ว; คู่ cross-split ไม่ถูกใช้.
2. รัน GA และเลือก genome ด้วย validation cost เท่านั้น.
3. หลังเลือก genome แล้วจึงคำนวณ baseline และ GA metrics บน held-out test.
4. ไม่ใช้ test cost, F1 หรือ metric ใดในการเลือก genome หรือ seed.

Baseline: **R2_manual** | GA experiment: **r3**

## Baseline บน held-out test

| Scenario | TP | FP | FN | REVIEW | Precision | Recall | F1 | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| custom | 6154 | 37 | 2183 | 462 | 0.9940 | 0.7013 | 0.8224 | 2562.24 |

## ผลราย seed (held-out test)

| Scenario | Seed | t_m | t_r | c_promote | c_demote | Validation cost | Test F1 | Test cost | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| custom | 7 | 0.9990 | 0.5343 | 0.9568 | 0.0000 | 554.18 | 0.8614 | 2228.34 | 0.57 |
| custom | 42 | 0.9990 | 0.5119 | 0.9565 | 0.2095 | 559.02 | 0.8615 | 2247.40 | 0.59 |
| custom | 123 | 0.9990 | 0.5296 | 0.9561 | 0.0000 | 554.18 | 0.8615 | 2228.32 | 0.63 |
| custom | 999 | 0.9921 | 0.5147 | 0.9561 | 0.2559 | 559.02 | 0.8615 | 2247.40 | 0.57 |
| custom | 2025 | 0.9985 | 0.5187 | 0.9561 | 0.3954 | 559.02 | 0.8615 | 2247.38 | 0.73 |

## สรุป mean ± std และการเลือก genome

### custom

- Test F1: 0.8615 ± 0.0000 (min 0.8614, max 0.8615)
- Test cost: 2239.77 ± 10.44 (min 2228.32, max 2247.40)
- เปลี่ยนจาก baseline (mean): F1 +4.752% | cost -12.586%
- Genome ที่เลือกจาก validation cost เท่านั้น: seed 7 [t_m=0.9990, t_r=0.5343, c_promote=0.9568, c_demote=0.0000] (val cost 554.18)

