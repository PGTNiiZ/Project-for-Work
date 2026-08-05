# Model Comparison

ชุดนี้ benchmark บน main multimodal run `image_context_r075_h20_s42` โดยใช้ feature 41 ตัว, split เดิม (`train/val/test`), isotonic calibration บน validation และเลือก threshold ที่ให้ F1 สูงสุดบน validation เช่นเดียวกับ pipeline หลัก

## Summary

| rank | model | test_ap | test_auc | test_f1 | test_precision | test_recall | threshold | why_selected | strengths | weaknesses | conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | gb | 0.9797 | 0.9737 | 0.9333 | 0.9618 | 0.9065 | 0.4600 | Boosted trees เป็นตัวหลักของ pipeline เดิมและเหมาะกับ tabular similarity features | จับ nonlinear interaction ได้ดี, ให้ feature importance, ผลมักนิ่งบนข้อมูลตาราง | เทรนช้ากว่า linear model และต้องคุม depth/learning rate | ตัวหลักที่แนะนำ: ชนะตาม validation composite และให้ test metrics สมดุลที่สุด |
| 2 | rf | 0.9764 | 0.9706 | 0.9289 | 0.9457 | 0.9127 | 0.4200 | Bagged trees เป็น baseline ensemble มาตรฐานสำหรับข้อมูลตาราง | ทน noise, ไม่ไวต่อ scaling, จับ interaction ได้ดี | probability มักไม่คมเท่า boosting และ model ใหญ่ | ตัวใกล้เคียง: คะแนนตามหลังน้อย ใช้เป็น alternative หรือ sensitivity check ได้ |
| 3 | mlp | 0.9734 | 0.9649 | 0.9255 | 0.9574 | 0.8956 | 0.4600 | Neural baseline เพื่อดูว่าการเพิ่มความยืดหยุ่นของโมเดลช่วยกับฟีเจอร์ชุดนี้หรือไม่ | เรียนรู้ interaction ซับซ้อนได้, เป็นตัวแทนของ neural family | จูนยาก, แปลผลยาก, ผลลัพธ์ไม่นิ่งเท่า tree ensemble บนข้อมูลตารางขนาดนี้ | neural reference: ใช้พิสูจน์ว่าการเพิ่มความซับซ้อนไม่ได้แปลว่าดีกว่าเสมอ |
| 4 | extra_trees | 0.9734 | 0.9664 | 0.9212 | 0.9427 | 0.9006 | 0.4400 | Randomized tree ensemble ใช้ดูว่าการสุ่ม split เพิ่ม generalization ได้หรือไม่ | เร็ว, ลด variance ได้ดี, มักแข็งแรงบน feature จำนวนปานกลาง | อาจ overfit noise บางส่วนและ probability ไม่ smooth | reference model: มีประโยชน์ในการเทียบ family ของโมเดล แต่ไม่ใช่ตัวหลัก |
| 5 | adaboost | 0.9363 | 0.9289 | 0.8762 | 0.9349 | 0.8245 | 0.3200 | Simple additive ensemble ใช้เป็น reference ของ boosting family ที่เบากว่า | ตีความง่ายกว่าบาง ensemble และใช้เป็น baseline ของ boosting | ไวต่อ noise/outlier และมักด้อยกว่า gradient boosting | reference model: มีประโยชน์ในการเทียบ family ของโมเดล แต่ไม่ใช่ตัวหลัก |
| 6 | logreg | 0.9422 | 0.9256 | 0.8629 | 0.8877 | 0.8395 | 0.3600 | Linear baseline ที่ตีความง่าย ใช้ตรวจว่าฟีเจอร์หลักแยกคลาสได้ด้วยเส้นแบ่งตรงหรือไม่ | เร็วมาก, อธิบาย coefficient ได้, ใช้เป็น baseline ที่เชื่อถือได้ | จับ nonlinear interaction และ threshold effect ได้จำกัด | baseline เชิงตีความ: ควรคงไว้ในเล่มเพื่อพิสูจน์ว่า nonlinear model ช่วยจริง |
| 7 | linear_svm | 0.9421 | 0.9255 | 0.8621 | 0.8867 | 0.8388 | 0.4000 | Large-margin linear model ใช้ดูว่าการแบ่งคลาสแบบเส้นขอบกว้างช่วยกว่าหรือด้อยกว่า logistic regression หรือไม่ | เร็ว, เหมาะกับ feature space เชิง similarity, เป็น baseline เชิง margin ที่ดี | ไม่ให้ probability ตรง ๆ, แปลผลยากกว่า logistic regression, จับ nonlinear interaction ไม่ได้ | reference model: มีประโยชน์ในการเทียบ family ของโมเดล แต่ไม่ใช่ตัวหลัก |

## Reading Guide

- `test_ap` ใช้ดูคุณภาพการจัดอันดับภายใต้ class imbalance
- `test_auc` ใช้ดูความสามารถในการแยก positive/negative โดยรวม
- `test_f1` ใช้ดูสมดุล precision/recall หลังเลือก threshold
- model ที่แนะนำควรดูทั้ง ranking, ความต่างของคะแนน, และความสามารถในการอธิบายต่ออาจารย์

## Recommendation

แนะนำให้ใช้ Gradient Boosting หรือ model ที่ชนะใน benchmark นี้เป็นตัวหลักของเล่ม โดยคง Logistic Regression
เป็น baseline เชิงตีความ และเก็บอีก 1-2 model ที่คะแนนใกล้เคียงเป็นตัวเปรียบเทียบ เพื่อพิสูจน์ว่าการเลือก
final model ไม่ได้มาจากความรู้สึก แต่ผ่านการทดลองบน feature และ split เดียวกันแล้ว
