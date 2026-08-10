# GA Automation: r3_smoke_v2

## ขั้นตอนการทดลอง

1. โหลด cache ที่แบ่ง entity-aware เป็น validation/test แล้ว; คู่ cross-split ไม่ถูกใช้.
2. รัน GA และเลือก genome ด้วย validation cost เท่านั้น.
3. หลังเลือก genome แล้วจึงคำนวณ baseline และ GA metrics บน held-out test.
4. ไม่ใช้ test cost, F1 หรือ metric ใดในการเลือก genome หรือ seed.

Baseline: **R2_manual** | GA experiment: **r3**

## Baseline บน held-out test

| Scenario | TP | FP | FN | REVIEW | Precision | Recall | F1 | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 5992 | 35 | 2162 | 687 | 0.9942 | 0.6828 | 0.8096 | 2350.74 |

## ผลราย seed (held-out test)

| Scenario | Seed | t_m | t_r | c_promote | c_demote | Validation cost | Test F1 | Test cost | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_current | 42 | 0.9990 | 0.5904 | 0.9577 | 0.3900 | 629.62 | 0.8543 | 2108.60 | 0.05 |

## สรุป mean ± std และการเลือก genome

### A_current

- Test F1: 0.8543 ± 0.0000 (min 0.8543, max 0.8543)
- Test cost: 2108.60 ± 0.00 (min 2108.60, max 2108.60)
- เปลี่ยนจาก baseline (mean): F1 +5.521% | cost -10.301%
- Genome ที่เลือกจาก validation cost เท่านั้น: seed 42 [t_m=0.9990, t_r=0.5904, c_promote=0.9577, c_demote=0.3900] (val cost 629.62)

