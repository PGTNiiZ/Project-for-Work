"""
Regenerate Embedding Paths
===========================
Match existing embedding files to profiles and update the CSV.
"""

import pandas as pd
import os
from glob import glob

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data/processed/profiles_with_embeddings.csv")
EMBEDDINGS_DIR = os.path.join(SCRIPT_DIR, "data/embeddings")

print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

print(f"Total profiles: {len(df)}")

# Get all embedding files
print(f"\nScanning embeddings directory: {EMBEDDINGS_DIR}")
embedding_files = glob(os.path.join(EMBEDDINGS_DIR, "*.npy"))
print(f"Found {len(embedding_files)} embedding files")

# Create a mapping: (profile_id, platform) -> embedding_path
embedding_map = {}

for emb_file in embedding_files:
    filename = os.path.basename(emb_file)
    # Format: {profile_id}_{platform}_embedding.npy
    # Example: ackleysuicide_twitter_embedding.npy
    
    if filename.endswith('_embedding.npy'):
        # Remove _embedding.npy suffix
        name_part = filename.replace('_embedding.npy', '')
        
        # Split by last underscore to get platform
        parts = name_part.rsplit('_', 1)
        if len(parts) == 2:
            profile_id, platform = parts
            embedding_map[(profile_id, platform)] = emb_file

print(f"Created mapping for {len(embedding_map)} embeddings")

# Update DataFrame
def get_embedding_path(row):
    """Get embedding path for a profile."""
    profile_id = row['profile_id']
    platform = row['platform']
    key = (profile_id, platform)
    return embedding_map.get(key, None)

print("\nMatching embeddings to profiles...")
df['embedding_path'] = df.apply(get_embedding_path, axis=1)

matched_count = df['embedding_path'].notna().sum()
print(f"Matched {matched_count} profiles to embeddings")

# Save updated CSV
print(f"\nSaving updated CSV to {CSV_PATH}...")
df.to_csv(CSV_PATH, index=False)

print("\n✓ Done!")
print("\nSample of matched profiles:")
print(df[df['embedding_path'].notna()][['profile_id', 'platform', 'embedding_path']].head(10))

# Statistics by platform
print("\nEmbeddings by platform:")
print(df[df['embedding_path'].notna()].groupby('platform').size())
