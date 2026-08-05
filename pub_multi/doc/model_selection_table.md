# Model Selection Table

ตารางนี้สรุปให้ตอบคำถาม 3 ข้อแบบตรง ๆ

- เราลองอะไรไปบ้าง
- ลองเพราะอะไร
- สุดท้ายเลือกตัวไหน และไม่เลือกตัวอื่นเพราะอะไร

## 1. Model Family History

| สิ่งที่ลอง | บทบาท | ทำไมเอามาลอง | ผลหลัก | ทำไมเลือก / ไม่เลือก |
| --- | --- | --- | --- | --- |
| `Exact match` | rule baseline | ใช้เป็น baseline ที่แม่นสูงและอธิบายง่ายที่สุด | precision สูงมาก แต่ recall ต่ำ ใช้เก็บคู่ที่มั่นใจสูงก่อนเข้า model | `เลือก` เป็นกติกา baseline ของระบบ แต่ `ไม่ใช่` model หลัก เพราะเก็บ true match ได้ไม่ครบ |
| `Logistic Regression` | linear baseline | ใช้ตรวจว่า feature หลักแยกคู่ match/non-match ได้ด้วยเส้นแบ่งตรงหรือไม่ | main benchmark: `AP 0.9422`, `AUC 0.9256`, `F1 0.8629` ; strict report point: `AP 0.8395`, `AUC 0.9973`, `F1 0.8951`, `P/R 0.9900 / 0.8169` | `เลือก` เป็น baseline เชิงตีความ และ `เลือกเพิ่ม` เป็น strict report reference เพราะให้ operating point ที่สมจริงกว่า strict ที่ precision = 1 แต่ `ไม่เลือก` เป็น final model หลักของเส้น multimodal เพราะ main benchmark ยังด้อยกว่า tree models |
| `Linear SVM` | linear margin baseline | ใช้เทียบกับ logistic regression ว่า large-margin linear model ช่วยไหม | main benchmark: `AP 0.9421`, `AUC 0.9255`, `F1 0.8621` | `เลือก` เป็น reference family แต่ `ไม่เลือก` เป็น final model เพราะผลแทบไม่ดีกว่า logistic และยังอธิบาย probability ยากกว่า |
| `AdaBoost` | lightweight boosting reference | ใช้ดูว่า boosting แบบง่ายช่วยได้แค่ไหน | main benchmark: `AP 0.9363`, `AUC 0.9289`, `F1 0.8762` | `ไม่เลือก` เพราะด้อยกว่า `gb` และ `rf` ชัด |
| `Extra Trees` | randomized tree ensemble | ใช้ดูว่า tree ensemble แบบสุ่ม split จะ generalize ดีไหม | main benchmark: `AP 0.9734`, `AUC 0.9664`, `F1 0.9212` ; strict no-leak: `AP 0.8409`, `F1 0.8992` | `เลือก` เป็น strong alternative/reference แต่ `ไม่เลือก` เป็น final model เพราะโดยรวมยังตาม `gb` หรือ `rf` เล็กน้อย |
| `Random Forest` | strong tree baseline | ใช้เป็น ensemble มาตรฐานของข้อมูล tabular | main benchmark: `AP 0.9764`, `AUC 0.9706`, `F1 0.9289` ; strict no-leak: `AP 0.8594`, `F1 0.8992` | `เลือก` เป็นตัวใกล้เคียงที่สุดและเป็น benchmark สำคัญ แต่ `ยังไม่เลือก` เป็น final model ของเส้น multimodal เพราะ main benchmark แพ้ `gb` เล็กน้อย |
| `MLP` | neural network reference | ใช้พิสูจน์ว่าถ้าเพิ่ม model complexity แบบ neural แล้วจะดีขึ้นจริงหรือไม่ | main benchmark: `AP 0.9734`, `AUC 0.9649`, `F1 0.9255` ; strict existing report: `AP 0.7344`, `F1 0.8296` | `เลือก` ให้มีในเล่มแน่ เพราะใช้ตอบคำถามเรื่อง neural network แต่ `ไม่เลือก` เป็น final model เพราะผลไม่ชนะ `gb`/`rf` และตีความยากกว่า |
| `Gradient Boosting` | final main model candidate | เหมาะกับ similarity features แบบตาราง และเป็นตัวหลักใน pipeline เดิม | main benchmark: `AP 0.9797`, `AUC 0.9737`, `F1 0.9333` ; strict no-leak rerun: `AP 0.8594`, `F1 0.8982` | `เลือก` เป็น final main model ของเส้น multimodal เพราะชนะ benchmark หลัก, ชนะ tuning ของ family ตัวเอง, และอธิบายได้ง่ายกว่า neural network |

## 2. Feature / Run History

ตรงนี้สำคัญ เพราะการเลือก final model ไม่ได้ดูแค่ family ของ model แต่ดูว่าใช้กับ feature set แบบไหนด้วย

| Run / Feature Set | ต่างจากตัวอื่นอย่างไร | ผลหลัก | ทำไมเลือก / ไม่เลือก |
| --- | --- | --- | --- |
| `text_attr_hybrid` | ใช้ text + attribute เป็น baseline ของสาย multimodal | `AP 0.9782`, `AUC 0.9715`, `F1 0.9330` | `เลือก` เป็น baseline ที่ดีมาก แต่ `ไม่เลือก` เป็น run หลัก เพราะยังด้อยกว่า image-enhanced runs เล็กน้อย |
| `image_stats` | เพิ่ม image statistics เข้าไป | `AP 0.9789`, `AUC 0.9731`, `F1 0.9340` | `เลือก` เป็นตัวใกล้เคียงมาก แต่ `ไม่เลือก` เป็น final run เพราะ composite score ตาม `image_context` เล็กน้อย |
| `image_context` | เพิ่ม image context / caption-cross signals ครบที่สุด | `AP 0.9789`, `AUC 0.9734`, `F1 0.9340` | `เลือก` เป็น final run เพราะได้ composite score สูงสุด และเป็น run ที่สะท้อนแนวคิด multimodal ครบที่สุด |

## 3. Tuning History

| สิ่งที่ปรับ | ค่าที่เด่น | ความหมาย |
| --- | --- | --- |
| `random_neg_ratio` | `0.75` | sampling negatives ระดับนี้ให้สมดุลระหว่างความยากของโจทย์กับการเรียนรู้ของโมเดล |
| `hard_neg_ratio` | `2.0 - 2.5` | ช่วยให้โมเดลเจอคู่ลบที่คล้ายกันจริงมากขึ้น |
| `gb` tuning ที่ดีที่สุด | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` | เป็น config ที่ชนะในการ tune ของ family `gb` |

## 4. Strict Operating Point ที่ใช้ในรายงาน

จุดนี้แยกจากการเลือก final main model โดยตรง กล่าวคือยังคงใช้ `image_context + gb` เป็นโมเดลหลักของเส้น multimodal เหมือนเดิม แต่สำหรับ “ผลอ้างอิงแบบ strict no-leak” ในรายงาน ควรใช้ operating point ที่สมจริงกว่าและไม่ตันที่ precision เท่ากับ 1

| strict candidate | ค่าใช้งาน | ผลที่ได้ | ใช้ในรายงานหรือไม่ | เหตุผล |
| --- | --- | --- | --- | --- |
| `rf + isotonic` | `threshold = 0.05` | `F1 0.8992`, `P 1.0000`, `R 0.8169`, `FP 0` | `ไม่ใช้เป็นจุดหลัก` | ตัวเลขดู conservative เกินไปและทำให้ precision ตันที่ 1 |
| `logreg + isotonic` | `threshold = 0.11` | `F1 0.8984`, `P 0.9980`, `R 0.8169`, `FP 2` | `ใช้ได้เป็นทางเลือก` | เปลี่ยนน้อยที่สุดจาก strict เดิม และทำให้ precision หลุดจาก 1 แบบนุ่มนวล |
| `logreg + sigmoid` | `threshold = 0.50` | `F1 0.8951`, `P 0.9900`, `R 0.8169`, `FP 10` | `เลือกใช้เป็น strict report point` | ยังรักษา recall เดิมไว้ แต่ได้ precision ที่ไม่ตันและอธิบาย trade-off ได้สมจริงกว่า |

ดังนั้น หากต้องมี strict operating point เพียงจุดเดียวในเล่ม ควรยึด `logreg + sigmoid @ 0.50` เป็นผลหลักของสาย strict สำหรับใช้ในบทประเมินผล ส่วน `rf + isotonic` และ `logreg + isotonic @ 0.11` ควรเก็บไว้เป็น reference หรือผลเสริมในภาคผนวก

## 5. Final Answer แบบสั้น

- ถามว่า “ลองหลาย model ไหม” คำตอบคือ `ลอง`
- ถามว่า “เลือกเพราะอะไร” คำตอบคือ `เลือกจาก benchmark + tuning + การตีความได้`
- ถามว่า “ต้องใช้ neural network ไหม” คำตอบคือ `ควรมีไว้เทียบ แต่ไม่จำเป็นต้องเป็น final model`
- final choice ของงานตอนนี้คือ `image_context + Gradient Boosting`

## 6. ประโยคที่ใช้เขียนในรายงานได้เลย

งานนี้ไม่ได้เลือกโมเดลจากคะแนนดีที่สุดเพียงตัวเดียวตั้งแต่ต้น แต่ดำเนินการเปรียบเทียบหลาย model families ได้แก่ linear models, tree ensembles และ neural network ภายใต้ feature set และ split ที่ควบคุมให้เทียบกันได้อย่างยุติธรรม ผลการทดลองชี้ว่า Gradient Boosting ให้สมดุลที่ดีที่สุดระหว่าง average precision, ROC-AUC, F1-score และความสามารถในการอธิบายผลลัพธ์ จึงถูกเลือกเป็นโมเดลหลักของ pipeline ขณะที่ MLP ถูกเก็บไว้เป็น comparative reference เพื่อพิสูจน์ว่าโมเดลเชิงลึกไม่ได้เหมาะสมที่สุดสำหรับข้อมูลลักษณะนี้เสมอไป

สำหรับสายประเมินแบบ strict no-leak ผู้วิจัยไม่ได้ใช้ operating point ที่ให้ precision เท่ากับ 1 เป็นผลอ้างอิงหลักเพียงอย่างเดียว เนื่องจากมีลักษณะ conservative มากเกินไปและอาจทำให้ผู้อ่านตีความว่าโมเดลสมบูรณ์แบบเกินจริง จึงเลือกใช้ผลของ `Logistic Regression` ที่ผ่าน `sigmoid calibration` และตั้งค่า threshold เท่ากับ `0.50` เป็น strict report point หลัก เนื่องจากยังคงรักษา recall ได้ที่ `0.8169` ขณะเดียวกันให้ precision `0.9900` ซึ่งสะท้อน trade-off ที่สมจริงและอธิบายได้ชัดเจนกว่า
