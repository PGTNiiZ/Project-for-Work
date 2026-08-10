# Experiment walkthrough notebooks

โฟลเดอร์นี้เก็บ notebook เพื่อเรียนรู้การทำงานของแต่ละ experiment โดยไม่แก้ไข
model, probability cache หรือผลลัพธ์เดิมใน `experiments/`.

## ลำดับที่แนะนำ

1. `00_shared_protocol/00_shared_protocol.ipynb` — split, metrics และ harness กลาง
2. `r0_baseline/R0_baseline.ipynb`
3. `r1_ga_redecision/R1_ga_redecision.ipynb`
4. `r2_minilm_feature/R2_minilm_feature.ipynb`
5. `r3_minilm_ga/R3_minilm_ga.ipynb`
6. `r4_bloom_manual/R4_bloom_manual.ipynb`
7. `r5_bloom_ga/R5_bloom_ga.ipynb`
8. `r6_bert_er/R6_bert_er.ipynb`
9. `gb_transformer_ga/GB_transformer_GA.ipynb`

แต่ละ notebook import และแสดง source code ที่ใช้จริงจาก root ของโปรเจกต์ด้วย
`inspect.getsource()` จึงไม่ใช่ pseudocode. Cell ที่รัน experiment เต็มถูก comment
ไว้ท้าย notebook เพื่อไม่ให้เผลอสร้าง artifact ใหม่ขณะศึกษา.

สร้าง notebook ใหม่จาก template ได้ด้วย:

```powershell
.\.venv\Scripts\python.exe experiments\walkthroughs\build_walkthrough_notebooks.py
```
