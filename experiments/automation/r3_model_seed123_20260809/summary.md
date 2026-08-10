# GA Automation: r3_model_seed123_20260809

## ขั้นตอนการทดลอง

1. โหลด cache ที่แบ่ง entity-aware เป็น validation/test แล้ว; คู่ cross-split ไม่ถูกใช้.
2. รัน GA และเลือก genome ด้วย validation cost เท่านั้น.
3. หลังเลือก genome แล้วจึงคำนวณ baseline และ GA metrics บน held-out test.
4. ไม่ใช้ test cost, F1 หรือ metric ใดในการเลือก genome หรือ seed.

Baseline: **R2_manual** | GA experiment: **r3**

## Baseline บน held-out test

| Scenario | TP | FP | FN | REVIEW | Precision | Recall | F1 | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 6597 | 65 | 2178 | 0 | 0.9902 | 0.7518 | 0.8547 | 2503.00 |

## ผลราย seed (held-out test)

| Scenario | Seed | t_m | t_r | c_promote | c_demote | Validation cost | Test F1 | Test cost | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 7 | 0.9990 | 0.5000 | 0.9526 | 0.5654 | 565.68 | 0.8620 | 1994.24 | 0.57 |
| A_current | 42 | 0.9990 | 0.5000 | 0.9504 | 0.4007 | 565.68 | 0.8620 | 2019.06 | 0.64 |
| A_current | 123 | 0.9960 | 0.5947 | 0.9518 | 0.5188 | 566.60 | 0.8619 | 2008.52 | 0.62 |
| A_current | 999 | 0.9959 | 0.6036 | 0.9529 | 0.2631 | 566.60 | 0.8621 | 1983.64 | 0.56 |
| A_current | 2025 | 0.9990 | 0.5000 | 0.9521 | 0.1892 | 565.68 | 0.8618 | 2014.16 | 0.67 |

## สรุป mean ± std และการเลือก genome

### A_current

- Test F1: 0.8620 ± 0.0001 (min 0.8618, max 0.8621)
- Test cost: 2003.92 ± 14.66 (min 1983.64, max 2019.06)
- เปลี่ยนจาก baseline (mean): F1 +0.849% | cost -19.939%
- Genome ที่เลือกจาก validation cost เท่านั้น: seed 7 [t_m=0.9990, t_r=0.5000, c_promote=0.9526, c_demote=0.5654] (val cost 565.68)

