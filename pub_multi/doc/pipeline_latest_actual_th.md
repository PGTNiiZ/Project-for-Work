# Pipeline ล่าสุดจริงที่สอดคล้องกับโค้ดปัจจุบัน

เอกสารนี้ใช้แทนภาพ plan เดิมในกรณีที่ต้องการให้รูปประกอบในรายงานตรงกับ implementation ล่าสุดของโครงการจริง โดยยึดเส้นหลักจากโค้ดปัจจุบัน ได้แก่ `preprocess_dataset.py`, `location_mapping_pipeline.py`, `create_normalized_db.py`, `stage8_pair_builder.py`, `stage9_features_pipeline_chunked.py`, `run_rebuilt_pipeline.py`, `run_multimodal_suite.py`, `stage10_13_training_pipeline.py`, `run_full_candidate_pipeline.py` และ `run_crm_entity_pipeline.py`

## รูปที่ 3.2 เวอร์ชันล่าสุดจริง: End-to-End Pipeline

คำอธิบายใต้ภาพที่ควรใช้  
แสดงลำดับการทำงานของระบบล่าสุดที่สอดคล้องกับโค้ดปัจจุบัน ตั้งแต่การนำเข้าข้อมูลดิบ การสร้าง artifact มาตรฐาน การสร้างคู่ข้อมูล การสร้าง feature matrix การเปรียบเทียบโมเดล การเลือก final model การคัด candidate pairs ในระดับ production ไปจนถึงการรวมเอนทิตีและการให้คะแนน lead ในระบบ CRM

```mermaid
flowchart TB
    classDef src fill:#ECEFF4,stroke:#4C566A,color:#111827,stroke-width:1px;
    classDef prep fill:#FEF3C7,stroke:#D97706,color:#111827,stroke-width:1px;
    classDef feat fill:#D1FAE5,stroke:#059669,color:#111827,stroke-width:1px;
    classDef model fill:#DBEAFE,stroke:#2563EB,color:#111827,stroke-width:1px;
    classDef prod fill:#FCE7F3,stroke:#DB2777,color:#111827,stroke-width:1px;
    classDef out fill:#EDE9FE,stroke:#7C3AED,color:#111827,stroke-width:1px;

    subgraph S1["Source Data and Normalization"]
        A["Raw LinkSocial JSON Files<br/>Twitter / Google+ / Instagram"]:::src
        B["clean_data/preprocess_dataset.py<br/>load_all_profiles + clean_dataframe"]:::prep
        C["all_profiles_cleaned.csv<br/>24,729 rows"]:::prep
        D["clean_data/location_mapping_pipeline.py<br/>location_mapping.csv<br/>4,589 raw locations / 462 mapped"]:::prep
        E["image_process/create_normalized_db.py<br/>normalized image artifacts<br/>images / metadata / quality / faces / captions / embeddings"]:::prep
        F["normalized_profiles_with_profile_id.csv<br/>36,807 rows"]:::prep
        A --> B --> C
        C --> D
        C --> E
        D --> F
        E --> F
    end

    subgraph S2["Training Data Preparation"]
        G["stage8_pair_builder.py<br/>positive + random negative + hard negative<br/>24,663,633 labeled pairs"]:::feat
        H["stage9_features_pipeline_chunked.py<br/>entity split + profile cache + TF-IDF + SBERT + chunked features"]:::feat
        I["Split artifacts<br/>train_profiles 15,848 / val_profiles 17,595 / test_profiles 3,364<br/>train_pairs 10,133,486 / val_pairs 4,511,311 / test_pairs 462,908"]:::feat
        J["Merged feature matrices<br/>train / val / test parquet"]:::feat
        F --> G --> H --> I --> J
    end

    subgraph S3["Model Development"]
        K["leakage_safe_experiment/run_experiment.py<br/>classical comparison<br/>logreg / gb / rf<br/>+ isotonic calibration + threshold search"]:::model
        L["stage7_8_rebuilt_experiment_hybrid/run_rebuilt_pipeline.py<br/>hybrid baseline with TF-IDF + SBERT + attribute features"]:::model
        M["stage7_13_multimodal_suite/run_multimodal_suite.py<br/>text_attr_hybrid / image_stats / image_context"]:::model
        N["stage10_13_training_pipeline.py<br/>IdentityMLP reference<br/>FocalLoss + Adam + CosineAnnealingLR<br/>+ isotonic calibration"]:::model
        O["Chosen main run<br/>image_context_r075_h20_s42<br/>Gradient Boosting + 41 features"]:::out
        F --> K
        F --> L --> M
        J --> N
        M --> O
    end

    subgraph S4["Production Retrieval and Scoring"]
        P["run_full_candidate_pipeline.py<br/>exact-first + multi-key blocking<br/>username_prefix3 / fullname_prefix3 / external_domain"]:::prod
        Q["Retrieval output<br/>all cross-platform pairs 449,149,239<br/>exact matches 12,403<br/>candidate pairs 2,073,842<br/>coverage 88.67%"]:::prod
        R["Candidate scoring<br/>load best_model + scaler + calibrator + feature_cols<br/>score candidates in chunks"]:::prod
        T["Decision tiers<br/>MATCH / REVIEW / NO_MATCH<br/>production thresholds 0.98 / 0.95"]:::prod
        F --> P --> Q --> R --> T
        O --> R
    end

    subgraph S5["CRM Workflow and Final Outputs"]
        U["run_crm_entity_pipeline.py<br/>match_decisions + review_queue"]:::prod
        V["Entity merge<br/>Union-Find transitive closure<br/>unified_profiles + profile_mapping"]:::out
        W["Lead scoring<br/>HOT / WARM / COLD"]:::out
        X["Final outputs<br/>review queue / final decisions / unified customer view / lead scores"]:::out
        T --> U --> V --> W --> X
    end
```

## รูปที่ 3.6 เวอร์ชันล่าสุดจริง: Model Development Pipeline

คำอธิบายใต้ภาพที่ควรใช้  
แสดงกระบวนการพัฒนาแบบจำลองล่าสุดของระบบ โดยแยกบทบาทของสาย classical comparison, multimodal suite และ neural reference ออกจากกันอย่างชัดเจน พร้อมแสดงจุดที่มีการทำ calibration, threshold selection และการเลือก final model สำหรับ production pipeline

```mermaid
flowchart TB
    classDef data fill:#FEF3C7,stroke:#D97706,color:#111827,stroke-width:1px;
    classDef exp fill:#DBEAFE,stroke:#2563EB,color:#111827,stroke-width:1px;
    classDef choose fill:#D1FAE5,stroke:#059669,color:#111827,stroke-width:1px;
    classDef final fill:#EDE9FE,stroke:#7C3AED,color:#111827,stroke-width:1px;

    A["Prepared training artifacts<br/>normalized profiles / labeled pairs / stage9 feature matrices"]:::data

    subgraph B1["Classical leakage-safe line"]
        C1["run_experiment.py<br/>train logreg / gb / rf"]:::exp
        C2["Validation ranking<br/>Average Precision + ROC-AUC"]:::exp
        C3["Isotonic calibration<br/>threshold search on validation"]:::exp
        C1 --> C2 --> C3
    end

    subgraph B2["Multimodal suite line"]
        D1["run_multimodal_suite.py<br/>text_attr_hybrid / image_stats / image_context"]:::exp
        D2["Shared cache<br/>SBERT embeddings / image profile features / caption embeddings"]:::exp
        D3["Leaderboard and suite report<br/>composite score ranking"]:::exp
        D2 --> D1 --> D3
    end

    subgraph B3["Neural reference line"]
        E1["stage10_13_training_pipeline.py<br/>IdentityMLP"]:::exp
        E2["undersample negatives + StandardScaler<br/>FocalLoss + Adam + CosineAnnealingLR"]:::exp
        E3["Isotonic calibration<br/>ECE before/after + threshold search"]:::exp
        E1 --> E2 --> E3
    end

    A --> C1
    A --> D2
    A --> E1

    F["Selection criteria<br/>performance + leakage robustness + interpretability + deployability"]:::choose
    C3 --> F
    D3 --> F
    E3 --> F

    G["Final main model<br/>Gradient Boosting on image_context<br/>run: image_context_r075_h20_s42"]:::final
    H["Production artifacts<br/>best_model.pkl / scaler.pkl / calibrator.pkl / feature_cols.pkl"]:::final
    F --> G --> H
```

## หมายเหตุสำคัญในการใช้รูปนี้ในรายงาน

1. รูปนี้สะท้อน implementation ล่าสุดจริง ไม่ใช่ roadmap เวอร์ชันแรก
2. หากต้องการอ้างการทดลองตามแผนเดิม ควรใช้คำว่า “แผนการพัฒนาเบื้องต้น” หรือ “initial implementation roadmap”
3. หากต้องการอธิบาย final pipeline ของงาน ควรใช้รูปชุดนี้แทนภาพเดิมที่ยังมี `LSH/SimHash`, `cross-modal attention` และ `OpenFace-heavy branch` เป็นแกนหลัก
