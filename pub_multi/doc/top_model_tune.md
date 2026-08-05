# Top Model Tuning

ตารางนี้เก็บเฉพาะค่าที่ดีที่สุดของแต่ละ family หลัง tune บน main multimodal run เดิม

## gb

- best config: `gb_2`
- params: `{"learning_rate": 0.05, "max_depth": 6, "n_estimators": 300}`
- val composite: `0.9615`
- test AP/AUC/F1: `0.9797 / 0.9737 / 0.9333`
- test Precision/Recall: `0.9618 / 0.9065`

## mlp

- best config: `mlp_1`
- params: `{"alpha": 0.0001, "hidden_layer_sizes": [128, 64], "learning_rate_init": 0.001}`
- val composite: `0.9540`
- test AP/AUC/F1: `0.9734 / 0.9649 / 0.9255`
- test Precision/Recall: `0.9574 / 0.8956`

## rf

- best config: `rf_4`
- params: `{"max_depth": 24, "min_samples_leaf": 2, "n_estimators": 500}`
- val composite: `0.9591`
- test AP/AUC/F1: `0.9767 / 0.9708 / 0.9307`
- test Precision/Recall: `0.9516 / 0.9106`
