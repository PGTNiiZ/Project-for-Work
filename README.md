# Project-for-Work
📁 Project-for-Work/
├── 📄 .gitignore
├── 📄 README.md
├── 📄 requirements.txt
│
├── 🐍 Python Scripts:
│   ├── advanced_embeddings.py
│   ├── create_normalized_db.py
│   ├── image_caption.py
│   ├── image_embeddings.py
│   ├── image_features.py
│   ├── load_dataset.py
│   ├── merge_image_data.py
│   └── preprocess_dataset.py
│
├── 📓 Notebooks:
│   ├── check_status.ipynb
│   └── dataset.ipynb
│
└── 📁 data/
    ├── 📄 combined_profiles.csv (12.4 MB)
    │
    ├── 📁 data/           # รูปภาพ profile (~48,886 ไฟล์)
    │
    ├── 📁 embeddings/     # Embedding vectors (.npy files)
    │   ├── clip/         # CLIP embeddings
    │   ├── face/         # ArcFace embeddings
    │   ├── fused/        # Fused embeddings
    │   └── *.npy         # Twitter embeddings
    │
    ├── 📁 features/
    │   ├── image_features.csv
    │   └── image_features.parquet
    │
    ├── 📁 final/
    │   ├── image_features_complete.csv
    │   ├── profiles_with_images.csv
    │   └── profiles_with_images.parquet
    │
    ├── 📁 images/         # รูปภาพ
    │
    ├── 📁 index/          # FAISS index files
    │
    ├── 📁 logs/
    │   └── pipeline_*.log (6 ไฟล์)
    │
    ├── 📁 manifest/       # Manifest data
    │
    ├── 📁 normalized/     # Normalized database (7 ไฟล์)
    │   ├── image_captions.csv/parquet
    │   ├── image_embeddings.csv/parquet
    │   ├── image_faces.csv/parquet
    │   ├── image_labels.csv/parquet
    │   ├── image_metadata.csv/parquet
    │   ├── image_quality.csv/parquet
    │   ├── images.csv/parquet
    │   └── image_database.sqlite
    │
    ├── 📁 output/
    │
    └── 📁 processed/      # Preprocessed data
        ├── all_profiles_cleaned.csv
        ├── df_googleplus.csv
        ├── df_instagram.csv
        ├── df_twitter.csv
        ├── instagram_googleplus_training.csv
        ├── twitter_googleplus_training.csv
        ├── twitter_instagram_training.csv
        ├── pairs_instagram_googleplus.csv
        ├── pairs_twitter_googleplus.csv
        ├── pairs_twitter_instagram.csv
        └── profiles_with_embeddings.csv