# STAGE MAP

- Stage 1: data prep -> `src/s01_prep.py`
- Stage 2: location normalization -> `src/s02_loc.py`
- Stage 3: normalized profile DB -> `src/s03_norm.py`
- Stage 4: image recovery -> `src/s04_img.py`
- Stage 5: pair construction -> `src/s05_pair.py`
- Stage 6: feature engineering -> `src/s06_feat.py`
- Stage 7: leak-safe baseline training -> `src/s07_base.py`
- Stage 8: multimodal training suite -> `src/s08_multi.py`
- Stage 9: full candidate retrieval + scoring -> `src/s09_full.py`
- Stage 10: CRM/entity export -> `src/s10_crm.py`
- Stage 11: strict MLP reference -> `src/s11_train.py`
- Stage 12: image pair helper -> `src/s12_imgpair.py`

หมายเหตุ: งานเดิมมีหลาย notebook/หลายเวอร์ชันปนกันอยู่ แพ็กเกจนี้เลือกเส้นทางหลักแบบ multimodal เพื่อให้เล่มมี version เดียวและตัวเลขไม่ขัดกันเอง
