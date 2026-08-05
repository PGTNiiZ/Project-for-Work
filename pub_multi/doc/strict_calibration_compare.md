# Strict Calibration Comparison

เอกสารนี้เปรียบเทียบการทำ calibration แบบ `isotonic` และ `sigmoid` บน strict no-leak setting เดียวกัน โดยใช้ model family และ feature set เดิม เพื่อดูว่าการปรับ calibration ทำให้ operating point สมจริงขึ้นหรือไม่

| model | calibrator | threshold | AP | AUC | F1 | Precision | Recall | FP | unique probs | หมายเหตุ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `extra_trees` | `isotonic` | 0.04 | 0.8409 | 0.9984 | 0.8992 | 1.0000 | 0.8169 | 0 | 4 | ยังเป็น conservative point |
| `extra_trees` | `sigmoid` | 0.14 | 0.8427 | 0.9981 | 0.8992 | 1.0000 | 0.8169 | 0 | 31 | ยังเป็น conservative point |
| `gb` | `isotonic` | 0.12 | 0.8594 | 0.9987 | 0.8982 | 1.0000 | 0.8152 | 0 | 6 | ยังเป็น conservative point |
| `gb` | `sigmoid` | 0.02 | 0.8594 | 0.9987 | 0.5868 | 0.4410 | 0.8766 | 1341 | 6 | recall สูงขึ้นแต่ precision ลดลงมาก |
| `logreg` | `isotonic` | 0.12 | 0.8392 | 0.9977 | 0.8982 | 1.0000 | 0.8152 | 0 | 6 | ยังเป็น conservative point |
| `logreg` | `sigmoid` | 0.50 | 0.8395 | 0.9973 | 0.8951 | 0.9900 | 0.8169 | 10 | 93 | สมจริงขึ้นและยังคุม FP ได้ดี |
| `rf` | `isotonic` | 0.05 | 0.8594 | 0.9987 | 0.8992 | 1.0000 | 0.8169 | 0 | 6 | ยังเป็น conservative point |
| `rf` | `sigmoid` | 0.48 | 0.8594 | 0.9987 | 0.2290 | 0.1293 | 1.0000 | 8129 | 155 | recall สูงขึ้นแต่ precision ลดลงมาก |

## ข้อสรุป

ผลที่รันได้จริงใน strict line นี้ชี้ว่า `sigmoid` ไม่ได้ช่วยทุกโมเดลเท่ากัน โดยกรณีที่เห็นผลชัดที่สุดคือ `logreg` ซึ่งเปลี่ยนจาก `precision = 1.0000` แบบ conservative ไปเป็น `precision = 0.9900` ที่ `FP = 10` โดยยังคง recall เท่าเดิมที่ `0.8169` จึงถือว่าเป็น operating point ที่สมจริงขึ้นและยังควบคุมความผิดพลาดได้ดี

ในทางตรงกันข้าม `rf` และ `gb` เมื่อใช้ `sigmoid` แล้ว threshold ที่เลือกจาก validation ทำให้ระบบทำนายกว้างขึ้นมากเกินไปจน false positives เพิ่มอย่างรุนแรง ส่วน `extra_trees` แม้ใช้ `sigmoid` แล้วจำนวน unique probabilities เพิ่มขึ้น แต่ operating point ที่เลือกก็ยังคงอยู่ที่ `precision = 1.0000` เหมือนเดิม ดังนั้นข้อสรุปของรอบนี้คือ การเปลี่ยน calibration อย่างเดียวไม่เพียงพอสำหรับทุก model family และไม่ควรคาดหวังว่า `sigmoid` จะทำให้ผลสมจริงขึ้นเสมอไป

หากต้องการผลที่สมจริงขึ้นสำหรับ strict setting แนวทางที่เหมาะสมกว่าคือรายงาน threshold sweep ควบคู่ไปกับผลหลัก และในกรณีที่ต้องเลือก operating point ใหม่จริง ควรเริ่มจาก `logreg + sigmoid` หรือเลือก threshold ของ `logreg + isotonic` ที่ต่ำลงเล็กน้อย เช่นบริเวณที่เริ่มมี `FP = 2-3` แทนการบังคับให้ `rf` หรือ `gb` หลุดจาก precision เท่ากับ 1 ด้วยการลด threshold อย่างแรงเกินไป
