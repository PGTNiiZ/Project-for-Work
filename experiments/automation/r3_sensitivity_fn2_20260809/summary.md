# GA Automation: r3_sensitivity_fn2_20260809

## ขั้นตอนการทดลอง

1. โหลด cache ที่แบ่ง entity-aware เป็น validation/test แล้ว; คู่ cross-split ไม่ถูกใช้.
2. รัน GA และเลือก genome ด้วย validation cost เท่านั้น.
3. หลังเลือก genome แล้วจึงคำนวณ baseline และ GA metrics บน held-out test.
4. ไม่ใช้ test cost, F1 หรือ metric ใดในการเลือก genome หรือ seed.

Baseline: **R2_manual** | GA experiment: **r3**

## Baseline บน held-out test

| Scenario | TP | FP | FN | REVIEW | Precision | Recall | F1 | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| custom | 6154 | 37 | 2183 | 462 | 0.9940 | 0.7013 | 0.8224 | 4560.24 |

## ผลราย seed (held-out test)

| Scenario | Seed | t_m | t_r | c_promote | c_demote | Validation cost | Test F1 | Test cost | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| custom | 7 | 0.9990 | 0.5000 | 0.9570 | 0.1008 | 1114.02 | 0.8614 | 3482.46 | 0.60 |
| custom | 42 | 0.9990 | 0.5007 | 0.9569 | 0.2385 | 1114.02 | 0.8614 | 3482.46 | 0.64 |
| custom | 123 | 0.9990 | 0.5000 | 0.9570 | 0.0000 | 1104.18 | 0.8614 | 3443.40 | 0.53 |
| custom | 999 | 0.9990 | 0.5082 | 0.9561 | 0.6143 | 1114.02 | 0.8615 | 3482.42 | 0.64 |
| custom | 2025 | 0.9990 | 0.5000 | 0.9562 | 0.2037 | 1114.02 | 0.8615 | 3482.44 | 0.58 |

## สรุป mean ± std และการเลือก genome

### custom

- Test F1: 0.8614 ± 0.0001 (min 0.8614, max 0.8615)
- Test cost: 3474.64 ± 17.46 (min 3443.40, max 3482.46)
- เปลี่ยนจาก baseline (mean): F1 +4.747% | cost -23.806%
- Genome ที่เลือกจาก validation cost เท่านั้น: seed 123 [t_m=0.9990, t_r=0.5000, c_promote=0.9570, c_demote=0.0000] (val cost 1104.18)

