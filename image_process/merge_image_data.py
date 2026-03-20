"""
Merge Profile Data with Image Features
======================================
Creates a comprehensive database for ML training by merging:
- Original profile data (24,729 profiles)
- Image features (metadata, blur, faces, labels, captions)
- Embeddings status

Output: CSV/Parquet ready for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input files
PROFILES_CSV = DATA_DIR / "processed" / "all_profiles_cleaned.csv"
IMAGE_FEATURES = DATA_DIR / "features" / "image_features.parquet"
CAPTION_DB = DATA_DIR / "output" / "manifest" / "caption_progress.sqlite"

# Output files
OUTPUT_DIR = DATA_DIR / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_PARQUET = OUTPUT_DIR / "profiles_with_images.parquet"
MERGED_CSV = OUTPUT_DIR / "profiles_with_images.csv"
IMAGE_ONLY_CSV = OUTPUT_DIR / "image_features_complete.csv"


def load_captions():
    """Load captions from SQLite database."""
    import sqlite3
    
    if not CAPTION_DB.exists():
        print("⚠ Caption DB not found")
        return pd.DataFrame()
    
    conn = sqlite3.connect(str(CAPTION_DB))
    
    try:
        captions_df = pd.read_sql("SELECT item_id, caption_en FROM captions WHERE status='done'", conn)
        print(f"✓ Loaded {len(captions_df)} captions")
        return captions_df
    except:
        return pd.DataFrame()
    finally:
        conn.close()


def create_merged_database():
    """Create comprehensive merged database."""
    
    print("=" * 60)
    print("CREATING MERGED IMAGE DATABASE")
    print("=" * 60)
    
    # ========== 1. Load Profile Data ==========
    print("\n1. Loading profile data...")
    if not PROFILES_CSV.exists():
        print(f"   ERROR: {PROFILES_CSV} not found!")
        return None
    
    profiles_df = pd.read_csv(PROFILES_CSV)
    print(f"   Loaded {len(profiles_df)} profiles")
    
    # ========== 2. Load Image Features ==========
    print("\n2. Loading image features...")
    if not IMAGE_FEATURES.exists():
        print(f"   ERROR: {IMAGE_FEATURES} not found!")
        return None
    
    features_df = pd.read_parquet(IMAGE_FEATURES)
    print(f"   Loaded {len(features_df)} image features")
    
    # ========== 3. Load Captions ==========
    print("\n3. Loading captions...")
    captions_df = load_captions()
    
    # ========== 4. Prepare merge key ==========
    print("\n4. Preparing merge keys...")
    
    # Create item_id from profiles (profile_id + platform + hash)
    # The image filename format: {profile_id}_{platform}_{url_hash}.jpg
    
    # First, let's extract profile_id and platform from image features item_id
    features_df['profile_id_extracted'] = features_df['item_id'].apply(
        lambda x: '_'.join(x.split('_')[:-2]) if len(x.split('_')) > 2 else x.split('_')[0]
    )
    features_df['platform_extracted'] = features_df['item_id'].apply(
        lambda x: x.split('_')[-2] if len(x.split('_')) > 2 else ''
    )
    
    # ========== 5. Merge features with captions ==========
    print("\n5. Merging features with captions...")
    if len(captions_df) > 0:
        features_df = features_df.merge(
            captions_df[['item_id', 'caption_en']], 
            on='item_id', 
            how='left'
        )
    else:
        features_df['caption_en'] = None
    
    # ========== 6. Create comprehensive image database ==========
    print("\n6. Creating image feature table...")
    
    # Select and rename columns for clarity
    image_columns = [
        'item_id',
        'image_path',
        'profile_id_extracted',
        'platform_extracted',
        # Metadata
        'width',
        'height',
        'aspect_ratio',
        'file_size_kb',
        # Quality
        'blur_score',
        'is_blurry',
        # Duplicate detection
        'phash',
        'duplicate_group_id',
        'is_duplicate',
        # Face detection
        'num_faces',
        'face_confidence',
        'face_area_ratio',
        'face_cluster_id',
        # Classification
        'top_label',
        'top_label_score',
        'label_scores',
        # Embeddings
        'has_clip_embedding',
        'has_face_embedding',
        # Caption
        'caption_en',
        # Status
        'status',
        'created_at'
    ]
    
    # Keep only existing columns
    existing_cols = [c for c in image_columns if c in features_df.columns]
    image_df = features_df[existing_cols].copy()
    
    # Rename for clarity
    image_df = image_df.rename(columns={
        'profile_id_extracted': 'profile_id',
        'platform_extracted': 'platform'
    })
    
    # ========== 7. Merge with original profiles ==========
    print("\n7. Merging with profile data...")
    
    # Create a lookup for image features
    image_lookup = image_df.set_index(['profile_id', 'platform'])
    
    # Merge profiles with image features
    merged_df = profiles_df.merge(
        image_df,
        on=['profile_id', 'platform'],
        how='left',
        suffixes=('', '_img')
    )
    
    print(f"   Merged: {len(merged_df)} profiles")
    print(f"   With images: {merged_df['item_id'].notna().sum()}")
    
    # ========== 8. Add image availability flag ==========
    merged_df['has_image'] = merged_df['item_id'].notna()
    
    # ========== 9. Save outputs ==========
    print("\n8. Saving outputs...")
    
    # Full merged dataset
    merged_df.to_parquet(MERGED_PARQUET, index=False)
    print(f"   ✓ {MERGED_PARQUET}")
    
    merged_df.to_csv(MERGED_CSV, index=False)
    print(f"   ✓ {MERGED_CSV}")
    
    # Image-only dataset (for image-specific ML)
    image_df.to_csv(IMAGE_ONLY_CSV, index=False)
    print(f"   ✓ {IMAGE_ONLY_CSV}")
    
    # ========== 10. Summary ==========
    print("\n" + "=" * 60)
    print("DATABASE CREATION COMPLETE!")
    print("=" * 60)
    
    print(f"\n📊 Statistics:")
    print(f"   Total profiles: {len(merged_df)}")
    print(f"   Profiles with images: {merged_df['has_image'].sum()}")
    print(f"   Profiles without images: {(~merged_df['has_image']).sum()}")
    
    if 'num_faces' in merged_df.columns:
        has_faces = merged_df['num_faces'].fillna(0) > 0
        print(f"   Images with faces: {has_faces.sum()}")
    
    if 'is_blurry' in merged_df.columns:
        print(f"   Blurry images: {merged_df['is_blurry'].sum()}")
    
    if 'is_duplicate' in merged_df.columns:
        print(f"   Duplicate images: {merged_df['is_duplicate'].sum()}")
    
    if 'caption_en' in merged_df.columns:
        has_caption = merged_df['caption_en'].notna()
        print(f"   With captions: {has_caption.sum()}")
    
    print(f"\n📁 Output files:")
    print(f"   - {MERGED_PARQUET} (full dataset for ML)")
    print(f"   - {MERGED_CSV} (full dataset readable)")
    print(f"   - {IMAGE_ONLY_CSV} (image features only)")
    
    print(f"\n📋 Columns in merged dataset ({len(merged_df.columns)}):")
    for i, col in enumerate(merged_df.columns):
        print(f"   {i+1:2}. {col}")
    
    return merged_df


if __name__ == "__main__":
    df = create_merged_database()
