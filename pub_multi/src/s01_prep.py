"""
Data Preprocessing Pipeline for Cross-Platform User Matching
=============================================================
This script performs:
1. Data Cleaning (NaN handling, text normalization)
2. Platform Splitting (separate DataFrames per platform)
3. Ground Truth Labeling (matching users across platforms)
"""

import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # Project-for-Work root
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "data" / "Dataset-LinkSocial"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================
# 1. DATA LOADING
# ============================================================

def load_all_profiles(base_path: str | Path = DEFAULT_DATASET_ROOT) -> pd.DataFrame:
    """Load all profiles from all 3 profile.data folders."""
    base_path = os.fspath(base_path)
    all_profiles = []
    profile_folders = ["1.profile.data", "2.profile.data", "3.profile.data"]
    
    for folder in profile_folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found, skipping...")
            continue
            
        print(f"Loading from {folder}...")
        
        for user_folder in os.listdir(folder_path):
            user_path = os.path.join(folder_path, user_folder)
            
            if os.path.isdir(user_path):
                for file in os.listdir(user_path):
                    if file.endswith('.json'):
                        file_path = os.path.join(user_path, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                data['source_folder'] = folder
                                data['user_folder'] = user_folder
                                
                                if 'twitter' in file.lower():
                                    data['platform'] = 'twitter'
                                elif 'instagram' in file.lower():
                                    data['platform'] = 'instagram'
                                elif 'google' in file.lower():
                                    data['platform'] = 'googleplus'
                                else:
                                    data['platform'] = 'unknown'
                                    
                                all_profiles.append(data)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
    
    return pd.DataFrame(all_profiles)


# ============================================================
# 2. DATA CLEANING
# ============================================================

def remove_emojis(text: str) -> str:
    """Remove emojis and special unicode characters from text."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Emoji pattern
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"  # enclosed characters
        "\U0001f926-\U0001f937"  # additional emoticons
        "\U00010000-\U0010ffff"  # supplementary planes
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove special chars, trim whitespace."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove emojis
    text = remove_emojis(text)
    # Convert to lowercase
    text = text.lower()
    # Remove @ symbol at the start (common in usernames)
    text = re.sub(r'^@', '', text)
    # Remove special characters except alphanumeric, spaces, underscores
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Trim
    text = text.strip()
    
    return text


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame:
    - Handle NaN values
    - Normalize text fields
    """
    df = df.copy()
    
    print("Starting data cleaning...")
    print(f"Original shape: {df.shape}")
    
    # ---- NaN Handling ----
    # For text fields, replace NaN with empty string
    text_columns = ['userName', 'fullName', 'bio', 'location', 'externalUrl', 'pictureURL']
    for col in text_columns:
        if col not in df.columns:
            df[col] = ''
        else:
            df[col] = df[col].fillna('')

    # For outputProfileName, use user_folder as backup
    if 'outputProfileName' in df.columns:
        if 'user_folder' in df.columns:
            df['outputProfileName'] = df['outputProfileName'].fillna(df['user_folder'])
        else:
            df['outputProfileName'] = df['outputProfileName'].fillna('')
    elif 'user_folder' in df.columns:
        df['outputProfileName'] = df['user_folder'].fillna('')
    else:
        df['outputProfileName'] = ''
    df['userName_clean'] = df['userName'].apply(normalize_text)
    df['fullName_clean'] = df['fullName'].apply(normalize_text)
    df['bio_clean'] = df['bio'].apply(lambda x: remove_emojis(str(x)) if pd.notna(x) else '')
    
    # Create a unified profile ID for ground truth matching
    # This uses outputProfileName which should be consistent across platforms
    df['profile_id'] = df['outputProfileName'].apply(normalize_text)
    
    print(f"Cleaned shape: {df.shape}")
    print(f"NaN counts after cleaning:")
    print(df[text_columns].isna().sum())
    
    return df


# ============================================================
# 3. PLATFORM SPLITTING
# ============================================================

def split_by_platform(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split DataFrame into separate DataFrames per platform.
    
    Returns:
        Dict with keys 'twitter', 'instagram', 'googleplus'
    """
    platforms = {}
    
    for platform in df['platform'].unique():
        platform_df = df[df['platform'] == platform].copy()
        platform_df = platform_df.reset_index(drop=True)
        platforms[platform] = platform_df
        print(f"{platform}: {len(platform_df)} profiles")
    
    return platforms


# ============================================================
# 4. GROUND TRUTH LABELING
# ============================================================

def create_ground_truth_pairs(
    df: pd.DataFrame,
    platform1: str = 'twitter',
    platform2: str = 'instagram'
) -> pd.DataFrame:
    """
    Create ground truth pairs for cross-platform matching.
    
    Uses 'profile_id' (derived from outputProfileName) to match users across platforms.
    
    Returns:
        DataFrame with columns:
        - profile_id: common identifier
        - {platform1}_idx: index in platform1 DataFrame
        - {platform2}_idx: index in platform2 DataFrame
        - {platform1}_userName, {platform2}_userName, etc.
    """
    # Filter by platforms
    df1 = df[df['platform'] == platform1].copy()
    df2 = df[df['platform'] == platform2].copy()
    
    # Find common profile IDs
    common_ids = set(df1['profile_id'].unique()) & set(df2['profile_id'].unique())
    print(f"Found {len(common_ids)} matching users between {platform1} and {platform2}")
    
    # Create pairs
    pairs = []
    for profile_id in common_ids:
        if not profile_id:  # Skip empty IDs
            continue
            
        row1 = df1[df1['profile_id'] == profile_id].iloc[0]
        row2 = df2[df2['profile_id'] == profile_id].iloc[0]
        
        pairs.append({
            'profile_id': profile_id,
            f'{platform1}_userName': row1['userName'],
            f'{platform1}_userName_clean': row1['userName_clean'],
            f'{platform1}_fullName': row1['fullName'],
            f'{platform1}_fullName_clean': row1['fullName_clean'],
            f'{platform1}_bio': row1['bio'],
            f'{platform1}_location': row1.get('location', ''),
            f'{platform2}_userName': row2['userName'],
            f'{platform2}_userName_clean': row2['userName_clean'],
            f'{platform2}_fullName': row2['fullName'],
            f'{platform2}_fullName_clean': row2['fullName_clean'],
            f'{platform2}_bio': row2['bio'],
            f'{platform2}_location': row2.get('location', ''),
            'is_match': 1  # These are true matches
        })
    
    pairs_df = pd.DataFrame(pairs)
    return pairs_df


def create_full_matching_dataset(
    df: pd.DataFrame,
    platform1: str = 'twitter',
    platform2: str = 'instagram',
    negative_ratio: float = 1.0
) -> pd.DataFrame:
    """
    Create a full matching dataset with both positive (matching) and negative (non-matching) pairs.
    
    Args:
        df: Cleaned DataFrame with all profiles
        platform1, platform2: Platforms to match
        negative_ratio: Ratio of negative samples to positive samples
    
    Returns:
        DataFrame with positive and negative pairs for training
    """
    # Get positive pairs
    positive_pairs = create_ground_truth_pairs(df, platform1, platform2)
    n_positive = len(positive_pairs)
    
    print(f"Created {n_positive} positive pairs")
    
    # Create negative pairs (non-matching users)
    df1 = df[df['platform'] == platform1].copy()
    df2 = df[df['platform'] == platform2].copy()
    
    n_negative = int(n_positive * negative_ratio)
    negative_pairs = []
    
    np.random.seed(42)  # For reproducibility
    
    # Get IDs that exist in both platforms (to ensure we pick non-matches)
    df1_ids = set(df1['profile_id'].unique())
    df2_ids = set(df2['profile_id'].unique())
    
    attempts = 0
    max_attempts = n_negative * 10
    
    while len(negative_pairs) < n_negative and attempts < max_attempts:
        attempts += 1
        
        # Random sample from each platform
        idx1 = np.random.randint(0, len(df1))
        idx2 = np.random.randint(0, len(df2))
        
        row1 = df1.iloc[idx1]
        row2 = df2.iloc[idx2]
        
        # Ensure they are NOT the same user
        if row1['profile_id'] != row2['profile_id']:
            negative_pairs.append({
                'profile_id': f"neg_{len(negative_pairs)}",
                f'{platform1}_userName': row1['userName'],
                f'{platform1}_userName_clean': row1['userName_clean'],
                f'{platform1}_fullName': row1['fullName'],
                f'{platform1}_fullName_clean': row1['fullName_clean'],
                f'{platform1}_bio': row1['bio'],
                f'{platform1}_location': row1.get('location', ''),
                f'{platform2}_userName': row2['userName'],
                f'{platform2}_userName_clean': row2['userName_clean'],
                f'{platform2}_fullName': row2['fullName'],
                f'{platform2}_fullName_clean': row2['fullName_clean'],
                f'{platform2}_bio': row2['bio'],
                f'{platform2}_location': row2.get('location', ''),
                'is_match': 0  # These are NOT matches
            })
    
    print(f"Created {len(negative_pairs)} negative pairs")
    
    negative_df = pd.DataFrame(negative_pairs)
    
    # Combine and shuffle
    full_dataset = pd.concat([positive_pairs, negative_df], ignore_index=True)
    full_dataset = full_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return full_dataset


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def run_preprocessing_pipeline(
    base_path: str | Path = DEFAULT_DATASET_ROOT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete preprocessing pipeline.
    
    Returns:
        Dictionary containing all processed DataFrames
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    df_raw = load_all_profiles(base_path)
    print(f"Loaded {len(df_raw)} total profiles")
    
    print("\n" + "=" * 60)
    print("STEP 2: Cleaning Data")
    print("=" * 60)

    if df_raw.empty:
        raise RuntimeError(
            f"No profiles were loaded from {base_path}. ตรวจสอบ path หรือข้อมูลในโฟลเดอร์ Dataset-LinkSocial"
        )

    df_clean = clean_dataframe(df_raw)

    print("\n" + "=" * 60)
    print("STEP 3: Splitting by Platform")
    print("=" * 60)
    platform_dfs = split_by_platform(df_clean)
    
    print("\n" + "=" * 60)
    print("STEP 4: Creating Ground Truth Pairs")
    print("=" * 60)
    
    # Create pairs for different platform combinations
    pairs = {}
    platform_combos = [
        ('twitter', 'instagram'),
        ('twitter', 'googleplus'),
        ('instagram', 'googleplus')
    ]
    
    for p1, p2 in platform_combos:
        if p1 in platform_dfs and p2 in platform_dfs:
            pair_key = f"{p1}_{p2}"
            pairs[pair_key] = create_ground_truth_pairs(df_clean, p1, p2)
            print(f"  {pair_key}: {len(pairs[pair_key])} pairs")
    
    print("\n" + "=" * 60)
    print("STEP 5: Creating Full Training Dataset")
    print("=" * 60)
    
    # Create full dataset with positive and negative pairs
    training_datasets = {}
    for p1, p2 in platform_combos:
        if p1 in platform_dfs and p2 in platform_dfs:
            key = f"{p1}_{p2}_training"
            training_datasets[key] = create_full_matching_dataset(df_clean, p1, p2, negative_ratio=1.0)
    
    print("\n" + "=" * 60)
    print("STEP 6: Saving Processed Data")
    print("=" * 60)
    
    # Save all cleaned data
    df_clean.to_csv(output_dir / "all_profiles_cleaned.csv", index=False)
    print(f"Saved: all_profiles_cleaned.csv")
    
    # Save platform-specific DataFrames
    for platform, pdf in platform_dfs.items():
        filename = f"df_{platform}.csv"
        pdf.to_csv(output_dir / filename, index=False)
        print(f"Saved: {filename} ({len(pdf)} rows)")
    
    # Save ground truth pairs
    for pair_key, pair_df in pairs.items():
        filename = f"pairs_{pair_key}.csv"
        pair_df.to_csv(output_dir / filename, index=False)
        print(f"Saved: {filename} ({len(pair_df)} rows)")
    
    # Save training datasets
    for key, tdf in training_datasets.items():
        filename = f"{key}.csv"
        tdf.to_csv(output_dir / filename, index=False)
        print(f"Saved: {filename} ({len(tdf)} rows)")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    
    # Return all DataFrames
    return {
        'raw': df_raw,
        'cleaned': df_clean,
        'platforms': platform_dfs,
        'pairs': pairs,
        'training': training_datasets
    }


# ============================================================
# USAGE
# ============================================================

if __name__ == "__main__":
    results = run_preprocessing_pipeline()
    
    print("\n\nSUMMARY:")
    print("-" * 40)
    print(f"Total profiles: {len(results['cleaned'])}")
    print(f"Platforms: {list(results['platforms'].keys())}")
    print(f"Ground truth pairs created:")
    for k, v in results['pairs'].items():
        print(f"  - {k}: {len(v)} pairs")
    print(f"\nTraining datasets created:")
    for k, v in results['training'].items():
        print(f"  - {k}: {len(v)} samples (pos+neg)")
