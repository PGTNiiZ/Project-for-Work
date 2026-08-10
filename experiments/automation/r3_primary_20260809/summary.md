# GA Automation: r3_primary_20260809

## ขั้นตอนการทดลอง

1. โหลด cache ที่แบ่ง entity-aware เป็น validation/test แล้ว; คู่ cross-split ไม่ถูกใช้.
2. รัน GA และเลือก genome ด้วย validation cost เท่านั้น.
3. หลังเลือก genome แล้วจึงคำนวณ baseline และ GA metrics บน held-out test.
4. ไม่ใช้ test cost, F1 หรือ metric ใดในการเลือก genome หรือ seed.

Baseline: **R2_manual** | GA experiment: **r3**

## Baseline บน held-out test

| Scenario | TP | FP | FN | REVIEW | Precision | Recall | F1 | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 6154 | 37 | 2183 | 462 | 0.9940 | 0.7013 | 0.8224 | 2377.24 |

## ผลราย seed (held-out test)

| Scenario | Seed | t_m | t_r | c_promote | c_demote | Validation cost | Test F1 | Test cost | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 7 | 0.9990 | 0.5222 | 0.9571 | 0.1217 | 559.02 | 0.8614 | 1917.40 | 0.68 |
| A_current | 42 | 0.9990 | 0.5245 | 0.9570 | 0.5122 | 559.02 | 0.8614 | 1917.40 | 0.61 |
| A_current | 123 | 0.9933 | 0.5322 | 0.9562 | 0.5188 | 559.02 | 0.8615 | 1917.38 | 0.61 |
| A_current | 999 | 0.9990 | 0.5262 | 0.9561 | 0.5480 | 559.02 | 0.8615 | 1917.38 | 0.54 |
| A_current | 2025 | 0.9959 | 0.5186 | 0.9561 | 0.2437 | 559.02 | 0.8615 | 1917.38 | 0.68 |

## สรุป mean ± std และการเลือก genome

### A_current

- Test F1: 0.8615 ± 0.0001 (min 0.8614, max 0.8615)
- Test cost: 1917.39 ± 0.01 (min 1917.38, max 1917.40)
- เปลี่ยนจาก baseline (mean): F1 +4.750% | cost -19.344%
- Genome ที่เลือกจาก validation cost เท่านั้น: seed 7 [t_m=0.9990, t_r=0.5222, c_promote=0.9571, c_demote=0.1217] (val cost 559.02)

