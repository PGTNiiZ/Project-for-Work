# MODEL

main run: `image_context_r075_h20_s42`

เหตุผล:
- AP สูงสุดใน suite: 0.9789
- AUC สูงสุดใน suite: 0.9734
- F1 สูงมากและใกล้กับตัวที่ดีที่สุดในเชิง F1
- ใช้ context จาก caption-to-text cross features ทำให้ narrative แบบ multimodal สมบูรณ์กว่า `image_stats`

best model ภายใน run คือ Gradient Boosting เพราะถูกเลือกเป็น best model ในทั้ง 3 experiments ของ suite และตีความได้ผ่าน feature importance
