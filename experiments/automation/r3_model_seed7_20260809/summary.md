# GA Automation: r3_model_seed7_20260809

## ขั้นตอนการทดลอง

1. โหลด cache ที่แบ่ง entity-aware เป็น validation/test แล้ว; คู่ cross-split ไม่ถูกใช้.
2. รัน GA และเลือก genome ด้วย validation cost เท่านั้น.
3. หลังเลือก genome แล้วจึงคำนวณ baseline และ GA metrics บน held-out test.
4. ไม่ใช้ test cost, F1 หรือ metric ใดในการเลือก genome หรือ seed.

Baseline: **R2_manual** | GA experiment: **r3**

## Baseline บน held-out test

| Scenario | TP | FP | FN | REVIEW | Precision | Recall | F1 | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 6438 | 53 | 2203 | 144 | 0.9918 | 0.7337 | 0.8434 | 2470.88 |

## ผลราย seed (held-out test)

| Scenario | Seed | t_m | t_r | c_promote | c_demote | Validation cost | Test F1 | Test cost | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 7 | 0.9900 | 0.5000 | 0.9506 | 0.3273 | 572.38 | 0.8624 | 2017.66 | 0.62 |
| A_current | 42 | 0.9990 | 0.5000 | 0.9508 | 0.4184 | 572.38 | 0.8623 | 2017.68 | 0.67 |
| A_current | 123 | 0.9990 | 0.5000 | 0.9511 | 0.0000 | 567.54 | 0.8624 | 1992.66 | 0.63 |
| A_current | 999 | 0.9964 | 0.5000 | 0.9505 | 0.5699 | 572.38 | 0.8623 | 2017.68 | 0.62 |
| A_current | 2025 | 0.9165 | 0.5000 | 0.9512 | 0.2615 | 586.38 | 0.8716 | 2092.62 | 0.57 |

## สรุป mean ± std และการเลือก genome

### A_current

- Test F1: 0.8642 ± 0.0041 (min 0.8623, max 0.8716)
- Test cost: 2027.66 ± 37.89 (min 1992.66, max 2092.62)
- เปลี่ยนจาก baseline (mean): F1 +2.466% | cost -17.938%
- Genome ที่เลือกจาก validation cost เท่านั้น: seed 123 [t_m=0.9990, t_r=0.5000, c_promote=0.9511, c_demote=0.0000] (val cost 567.54)

