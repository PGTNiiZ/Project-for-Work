# GA Automation: r1_20260808_200954

## ขั้นตอนการทดลอง

1. โหลด cache ที่แบ่ง entity-aware เป็น validation/test แล้ว; คู่ cross-split ไม่ถูกใช้.
2. รัน GA และเลือก genome ด้วย validation cost เท่านั้น.
3. หลังเลือก genome แล้วจึงคำนวณ baseline และ GA metrics บน held-out test.
4. ไม่ใช้ test cost, F1 หรือ metric ใดในการเลือก genome หรือ seed.

Baseline: **R0_production** | GA experiment: **r1**

## Baseline บน held-out test

| Scenario | TP | FP | FN | REVIEW | Precision | Recall | F1 | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 5864 | 88 | 1689 | 8783 | 0.9852 | 0.6683 | 0.7964 | 2304.66 |

## ผลราย seed (held-out test)

| Scenario | Seed | t_m | t_r | c_promote | c_demote | Validation cost | Test F1 | Test cost | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 7 | 0.9971 | 0.9639 | 0.9875 | 0.0928 | 5243.78 | 0.8292 | 2062.34 | 11.24 |
| A_current | 42 | 0.9990 | 0.9642 | 0.9875 | 0.5052 | 5243.72 | 0.8292 | 2062.32 | 12.44 |
| A_current | 123 | 0.9990 | 0.9641 | 0.9867 | 0.5700 | 5243.68 | 0.8292 | 2062.32 | 11.70 |
| A_current | 999 | 0.9990 | 0.9642 | 0.9873 | 0.3548 | 5243.74 | 0.8292 | 2062.32 | 11.98 |
| A_current | 2025 | 0.9990 | 0.9640 | 0.9869 | 0.5124 | 5243.74 | 0.8292 | 2062.32 | 11.02 |

## สรุป mean ± std และการเลือก genome

### A_current

- Test F1: 0.8292 ± 0.0000 (min 0.8292, max 0.8292)
- Test cost: 2062.32 ± 0.01 (min 2062.32, max 2062.34)
- เปลี่ยนจาก baseline (mean): F1 +4.119% | cost -10.515%
- Genome ที่เลือกจาก validation cost เท่านั้น: seed 123 [t_m=0.9990, t_r=0.9641, c_promote=0.9867, c_demote=0.5700] (val cost 5243.68)

