"""
Normalized Image Database
=========================
Creates normalized database structure with multiple tables linked by item_id.

Tables:
1. images          - Core image info (item_id, path, profile_id, platform)
2. image_metadata  - Dimensions (width, height, aspect_ratio, file_size)
3. image_quality   - Quality metrics (blur_score, is_blurry, phash, is_duplicate)
4. image_faces     - Face detection (num_faces, confidence, area_ratio, cluster_id)
5. image_labels    - CLIP classification (top_label, scores)
6. image_captions  - BLIP captions (caption_en)
7. image_embeddings - Embedding references (has_clip, has_face, paths)

All tables linked by: item_id (partition key)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sqlite3

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input
IMAGE_FEATURES = DATA_DIR / "features" / "image_features.parquet"
CAPTION_DB = DATA_DIR / "output" / "manifest" / "caption_progress.sqlite"
EMBEDDINGS_CLIP = DATA_DIR / "embeddings" / "clip"
EMBEDDINGS_FACE = DATA_DIR / "embeddings" / "face"

# Output
OUTPUT_DIR = DATA_DIR / "normalized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_normalized_tables():
    """Create normalized database with separate tables."""
    
    print("=" * 60)
    print("CREATING NORMALIZED IMAGE DATABASE")
    print("=" * 60)
    
    # Load features
    print("\n1. Loading image features...")
    if not IMAGE_FEATURES.exists():
        print(f"   ERROR: {IMAGE_FEATURES} not found!")
        return
    
    df = pd.read_parquet(IMAGE_FEATURES)
    print(f"   Loaded {len(df)} images")
    
    # Extract profile_id and platform from item_id
    df['profile_id'] = df['item_id'].apply(
        lambda x: '_'.join(x.split('_')[:-2]) if len(x.split('_')) > 2 else x.split('_')[0]
    )
    df['platform'] = df['item_id'].apply(
        lambda x: x.split('_')[-2] if len(x.split('_')) > 2 else ''
    )
    
    # Load captions
    print("\n2. Loading captions...")
    captions_df = pd.DataFrame()
    if CAPTION_DB.exists():
        conn = sqlite3.connect(str(CAPTION_DB))
        try:
            captions_df = pd.read_sql(
                "SELECT item_id, caption_en FROM captions WHERE status='done'", 
                conn
            )
            print(f"   Loaded {len(captions_df)} captions")
        except:
            pass
        finally:
            conn.close()
    
    # ========== CREATE NORMALIZED TABLES ==========
    print("\n3. Creating normalized tables...")
    
    # ----- Table 1: images (core) -----
    images = df[['item_id', 'image_path', 'profile_id', 'platform']].copy()
    images['created_at'] = df['created_at']
    
    # ----- Table 2: image_metadata -----
    image_metadata = df[['item_id', 'width', 'height', 'aspect_ratio', 'file_size_kb']].copy()
    
    # ----- Table 3: image_quality -----
    image_quality = df[['item_id', 'blur_score', 'is_blurry', 'phash', 
                        'duplicate_group_id', 'is_duplicate']].copy()
    
    # ----- Table 4: image_faces -----
    image_faces = df[['item_id', 'num_faces', 'face_confidence', 
                      'face_area_ratio', 'face_cluster_id']].copy()
    
    # ----- Table 5: image_labels -----
    image_labels = df[['item_id', 'top_label', 'top_label_score', 'label_scores']].copy()
    
    # ----- Table 6: image_captions -----
    if len(captions_df) > 0:
        image_captions = captions_df[['item_id', 'caption_en']].copy()
    else:
        image_captions = pd.DataFrame(columns=['item_id', 'caption_en'])
    
    # ----- Table 7: image_embeddings -----
    image_embeddings = df[['item_id', 'has_clip_embedding', 'has_face_embedding']].copy()
    image_embeddings['clip_embedding_path'] = image_embeddings['item_id'].apply(
        lambda x: str(EMBEDDINGS_CLIP / f"{x}.npy") if (EMBEDDINGS_CLIP / f"{x}.npy").exists() else None
    )
    image_embeddings['face_embedding_path'] = image_embeddings['item_id'].apply(
        lambda x: str(EMBEDDINGS_FACE / f"{x}.npy") if (EMBEDDINGS_FACE / f"{x}.npy").exists() else None
    )
    
    # ========== SAVE TABLES ==========
    print("\n4. Saving normalized tables...")
    
    tables = {
        'images': images,
        'image_metadata': image_metadata,
        'image_quality': image_quality,
        'image_faces': image_faces,
        'image_labels': image_labels,
        'image_captions': image_captions,
        'image_embeddings': image_embeddings
    }
    
    for name, table in tables.items():
        # Save as Parquet
        parquet_path = OUTPUT_DIR / f"{name}.parquet"
        table.to_parquet(parquet_path, index=False)
        
        # Save as CSV
        csv_path = OUTPUT_DIR / f"{name}.csv"
        table.to_csv(csv_path, index=False)
        
        print(f"   ✓ {name}: {len(table)} rows, {len(table.columns)} cols")
    
    # ========== CREATE SQLite DATABASE ==========
    print("\n5. Creating SQLite database...")
    
    db_path = OUTPUT_DIR / "image_database.sqlite"
    conn = sqlite3.connect(str(db_path))
    
    for name, table in tables.items():
        table.to_sql(name, conn, if_exists='replace', index=False)
        
        # Create index on item_id
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_item_id ON {name}(item_id)")
    
    conn.commit()
    conn.close()
    print(f"   ✓ {db_path}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("NORMALIZED DATABASE COMPLETE!")
    print("=" * 60)
    
    print("\n📁 Output files:")
    print(f"   {OUTPUT_DIR}/")
    for name in tables.keys():
        print(f"   ├── {name}.parquet")
        print(f"   ├── {name}.csv")
    print(f"   └── image_database.sqlite")
    
    print("\n📋 Table Schema:")
    print("""
    ┌─────────────────────┐      ┌──────────────────────┐
    │      images         │      │   image_metadata     │
    ├─────────────────────┤      ├──────────────────────┤
    │ item_id (PK)   ─────┼──────│ item_id (FK)         │
    │ image_path          │      │ width                │
    │ profile_id          │      │ height               │
    │ platform            │      │ aspect_ratio         │
    │ created_at          │      │ file_size_kb         │
    └─────────────────────┘      └──────────────────────┘
              │
              │ item_id
              ▼
    ┌─────────────────────┐      ┌──────────────────────┐
    │   image_quality     │      │    image_faces       │
    ├─────────────────────┤      ├──────────────────────┤
    │ item_id (FK)        │      │ item_id (FK)         │
    │ blur_score          │      │ num_faces            │
    │ is_blurry           │      │ face_confidence      │
    │ phash               │      │ face_area_ratio      │
    │ duplicate_group_id  │      │ face_cluster_id      │
    │ is_duplicate        │      └──────────────────────┘
    └─────────────────────┘
              │
              │ item_id
              ▼
    ┌─────────────────────┐      ┌──────────────────────┐
    │   image_labels      │      │  image_captions      │
    ├─────────────────────┤      ├──────────────────────┤
    │ item_id (FK)        │      │ item_id (FK)         │
    │ top_label           │      │ caption_en           │
    │ top_label_score     │      └──────────────────────┘
    │ label_scores (JSON) │
    └─────────────────────┘      ┌──────────────────────┐
              │                  │  image_embeddings    │
              │                  ├──────────────────────┤
              └──────────────────│ item_id (FK)         │
                                 │ has_clip_embedding   │
                                 │ has_face_embedding   │
                                 │ clip_embedding_path  │
                                 │ face_embedding_path  │
                                 └──────────────────────┘
    """)
    
    print("\n🔗 Partition Key: item_id")
    print("   Use JOIN on item_id to combine tables") 
    
    print("\n📝 Example Query (SQLite):")
    print("""
    SELECT 
        i.item_id, i.profile_id, i.platform,
        m.width, m.height,
        q.blur_score, q.is_duplicate,
        f.num_faces, f.face_cluster_id,
        l.top_label,
        c.caption_en
    FROM images i
    LEFT JOIN image_metadata m ON i.item_id = m.item_id
    LEFT JOIN image_quality q ON i.item_id = q.item_id
    LEFT JOIN image_faces f ON i.item_id = f.item_id
    LEFT JOIN image_labels l ON i.item_id = l.item_id
    LEFT JOIN image_captions c ON i.item_id = c.item_id
    WHERE q.is_duplicate = 0 AND q.is_blurry = 0
    """)
    
    return tables


if __name__ == "__main__":
    tables = create_normalized_tables()
