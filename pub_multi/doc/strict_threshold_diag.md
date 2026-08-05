# Strict Threshold Diagnostic

เอกสารนี้ใช้สำหรับอธิบายว่าทำไม strict rerun หลายโมเดลจึงมี precision เท่ากับ `1.0000` และถ้าต้องการ operating point ที่สมจริงขึ้นควรขยับ threshold ไปอย่างไร

| model | current threshold | current P | current R | current FP | first threshold with P<1 (test diag) | P | R | FP | unique calibrated probs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `extra_trees` | 0.04 | 1.0000 | 0.8169 | 0 | 0.02 | 0.1311 | 0.9983 | 7986 | 4 |
| `rf` | 0.05 | 1.0000 | 0.8169 | 0 | 0.04 | 0.4410 | 0.8766 | 1341 | 6 |
| `gb` | 0.12 | 1.0000 | 0.8152 | 0 | 0.04 | 0.4410 | 0.8766 | 1341 | 6 |
| `logreg` | 0.12 | 1.0000 | 0.8152 | 0 | 0.11 | 0.9980 | 0.8169 | 2 | 6 |

## ข้อสังเกต

ผลวิเคราะห์ชี้ว่าคะแนน calibrated probability ของ strict rerun มีค่าอยู่เพียงไม่กี่ระดับเท่านั้น ทำให้ threshold ขยับเพียงเล็กน้อยก็อาจทำให้จำนวน false positives กระโดดขึ้นทันที แทนที่จะค่อย ๆ เปลี่ยนอย่างต่อเนื่อง ปัญหานี้สอดคล้องกับการใช้ isotonic calibration บน validation set ที่แยกคู่บวกและคู่ลบได้ง่ายมาก จึงเกิด probability แบบ stepwise และทำให้ operating point ที่ได้มีลักษณะ conservative มาก

ดังนั้น หากต้องการผลที่สมจริงขึ้น ไม่ควรรายงานเฉพาะจุดที่ precision เท่ากับ 1 เพียงจุดเดียว แต่ควรรายงาน threshold sweep ร่วมด้วย และพิจารณาเปลี่ยน calibration strategy หรือใช้ validation set ที่ยากขึ้นเพื่อให้ probability distribution ละเอียดและสะท้อนความไม่แน่นอนมากขึ้น