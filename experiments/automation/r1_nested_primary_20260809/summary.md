# GA Automation: r1_nested_primary_20260809

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
| A_current | 7 | 0.9990 | 0.8312 | 0.9561 | 0.3282 | 542.28 | 0.8564 | 2329.80 | 0.37 |
| A_current | 42 | 0.9990 | 0.8090 | 0.9561 | 0.6464 | 541.44 | 0.8564 | 2329.66 | 0.35 |
| A_current | 123 | 0.9990 | 0.7910 | 0.9567 | 0.6422 | 541.62 | 0.8563 | 2330.26 | 0.37 |
| A_current | 999 | 0.9990 | 0.8313 | 0.9570 | 0.6459 | 541.42 | 0.8563 | 2329.48 | 0.37 |
| A_current | 2025 | 0.9990 | 0.8180 | 0.9570 | 0.6464 | 541.44 | 0.8563 | 2329.58 | 0.33 |

## สรุป mean ± std และการเลือก genome

### A_current

- Test F1: 0.8563 ± 0.0001 (min 0.8563, max 0.8564)
- Test cost: 2329.76 ± 0.31 (min 2329.48, max 2330.26)
- เปลี่ยนจาก baseline (mean): F1 +7.526% | cost +1.089%
- Genome ที่เลือกจาก validation cost เท่านั้น: seed 999 [t_m=0.9990, t_r=0.8313, c_promote=0.9570, c_demote=0.6459] (val cost 541.42)

