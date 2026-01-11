"""
Image Feature Extraction Pipeline
==================================
Comprehensive image analysis with GPU acceleration.

Processing Steps (Per Image):
1. Load image
2. Metadata: width, height, aspect_ratio, file_size
3. Blur score (Laplacian variance)
4. pHash (perceptual hash for duplicate detection)
5. Face detection: num_faces, confidence, area_ratio
6. CLIP embedding (reuse existing)
7. ArcFace embedding (reuse existing)
8. CLIP zero-shot labels

Post-Processing:
9. Face clustering (DBSCAN)
10. Duplicate detection (pHash similarity)

Output: Parquet file for easy ML training
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from tqdm import tqdm
import torch
import json
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GPU CONFIGURATION
# ============================================================

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

if CUDA_AVAILABLE:
    print(f"🚀 CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ Running on CPU")

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"

DIRS = {
    'images': DATA_DIR / "images",
    'embeddings_clip': DATA_DIR / "embeddings" / "clip",
    'embeddings_face': DATA_DIR / "embeddings" / "face",
    'features': DATA_DIR / "features",
}

FILES = {
    'output_parquet': DIRS['features'] / "image_features.parquet",
    'output_csv': DIRS['features'] / "image_features.csv",
}

# Blur threshold (lower = more blurry)
BLUR_THRESHOLD = 100

# Zero-shot labels for classification
ZERO_SHOT_LABELS = [
    "a photo of a person",
    "a selfie photo",
    "a professional headshot",
    "a company logo",
    "a cartoon or avatar",
    "a group photo of people",
    "a landscape or scenery",
    "an animal photo",
    "text or screenshot",
    "an abstract or artistic image"
]

# Short label names for output
LABEL_SHORT_NAMES = [
    "person", "selfie", "headshot", "logo", "cartoon",
    "group", "landscape", "animal", "screenshot", "abstract"
]


# ============================================================
# 1. SETUP
# ============================================================

def setup_directories():
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
    print("✓ Directories ready")


# ============================================================
# 2. IMAGE METADATA EXTRACTOR
# ============================================================

def extract_metadata(image_path: Path) -> Dict:
    """Extract basic image metadata."""
    try:
        file_size_kb = image_path.stat().st_size / 1024
        
        with Image.open(image_path) as img:
            width, height = img.size
            aspect_ratio = width / height if height > 0 else 0
            
        return {
            'width': width,
            'height': height,
            'aspect_ratio': round(aspect_ratio, 3),
            'file_size_kb': round(file_size_kb, 2)
        }
    except Exception as e:
        return {'width': 0, 'height': 0, 'aspect_ratio': 0, 'file_size_kb': 0}


# ============================================================
# 3. BLUR DETECTION
# ============================================================

def calculate_blur_score(image_path: Path) -> Tuple[float, bool]:
    """
    Calculate blur score using Laplacian variance.
    Higher = sharper, Lower = blurry
    """
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0, True
        
        # Resize for consistent measurement
        img = cv2.resize(img, (256, 256))
        
        # Laplacian variance
        blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
        is_blurry = blur_score < BLUR_THRESHOLD
        
        return round(blur_score, 2), is_blurry
    except:
        return 0.0, True


# ============================================================
# 4. PERCEPTUAL HASH (pHash)
# ============================================================

def calculate_phash(image_path: Path) -> str:
    """Calculate perceptual hash for duplicate detection."""
    try:
        import imagehash
        with Image.open(image_path) as img:
            phash = str(imagehash.phash(img))
        return phash
    except:
        return ""


# ============================================================
# 5. FACE DETECTION (GPU)
# ============================================================

class FaceDetector:
    """Face detection using DeepFace/RetinaFace."""
    
    def __init__(self):
        self.detector = None
        self.available = False
        
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            self.available = True
            print("✓ Face detector ready")
        except ImportError:
            print("⚠ DeepFace not available")
    
    def detect(self, image_path: str) -> Dict:
        """Detect faces and return stats."""
        if not self.available:
            return {'num_faces': 0, 'face_confidence': 0, 'face_area_ratio': 0}
        
        try:
            faces = self.DeepFace.extract_faces(
                img_path=str(image_path),
                detector_backend="opencv",
                enforce_detection=False
            )
            
            if not faces:
                return {'num_faces': 0, 'face_confidence': 0, 'face_area_ratio': 0}
            
            num_faces = len(faces)
            
            # Get image dimensions
            with Image.open(image_path) as img:
                img_area = img.size[0] * img.size[1]
            
            # Calculate total face area ratio
            total_face_area = 0
            total_confidence = 0
            
            for face in faces:
                region = face.get('facial_area', {})
                w = region.get('w', 0)
                h = region.get('h', 0)
                total_face_area += w * h
                total_confidence += face.get('confidence', 0)
            
            face_area_ratio = total_face_area / img_area if img_area > 0 else 0
            avg_confidence = total_confidence / num_faces if num_faces > 0 else 0
            
            return {
                'num_faces': num_faces,
                'face_confidence': round(avg_confidence, 3),
                'face_area_ratio': round(face_area_ratio, 3)
            }
            
        except:
            return {'num_faces': 0, 'face_confidence': 0, 'face_area_ratio': 0}


# ============================================================
# 6-7. EMBEDDING LOADERS (Reuse existing)
# ============================================================

def load_existing_embedding(item_id: str, emb_type: str = 'clip') -> Optional[np.ndarray]:
    """Load existing embedding from .npy file."""
    if emb_type == 'clip':
        emb_path = DIRS['embeddings_clip'] / f"{item_id}.npy"
    else:
        emb_path = DIRS['embeddings_face'] / f"{item_id}.npy"
    
    if emb_path.exists():
        return np.load(emb_path)
    return None


# ============================================================
# 8. CLIP ZERO-SHOT CLASSIFICATION (GPU)
# ============================================================

class CLIPZeroShot:
    """CLIP zero-shot classification with GPU support."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.available = False
        self.text_features = None
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            print("Loading CLIP for zero-shot...")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(DEVICE).eval()
            
            if CUDA_AVAILABLE:
                self.model = self.model.half()
            
            # Pre-compute text features for labels
            self._precompute_text_features()
            
            self.available = True
            print(f"✓ CLIP zero-shot ready (GPU: {CUDA_AVAILABLE})")
            
        except Exception as e:
            print(f"⚠ CLIP not available: {e}")
    
    def _precompute_text_features(self):
        """Pre-compute text embeddings for labels."""
        inputs = self.processor(text=ZERO_SHOT_LABELS, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            self.text_features = self.model.get_text_features(**inputs)
            self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
    
    def classify(self, image_path: str) -> Dict:
        """Classify image with zero-shot labels."""
        if not self.available:
            return {'top_label': '', 'top_score': 0, 'label_scores': {}}
        
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            if CUDA_AVAILABLE:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarities
                similarities = (image_features @ self.text_features.T).squeeze()
                probs = similarities.softmax(dim=-1).cpu().numpy()
            
            # Create label scores dict
            label_scores = {LABEL_SHORT_NAMES[i]: round(float(probs[i]), 3) 
                          for i in range(len(LABEL_SHORT_NAMES))}
            
            top_idx = probs.argmax()
            
            return {
                'top_label': LABEL_SHORT_NAMES[top_idx],
                'top_score': round(float(probs[top_idx]), 3),
                'label_scores': label_scores
            }
            
        except:
            return {'top_label': '', 'top_score': 0, 'label_scores': {}}


# ============================================================
# 9. FACE CLUSTERING (DBSCAN)
# ============================================================

def cluster_faces(embeddings: np.ndarray, item_ids: List[str], 
                  eps: float = 0.5, min_samples: int = 2) -> Dict[str, int]:
    """
    Cluster face embeddings using DBSCAN.
    Returns mapping of item_id -> cluster_id
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import normalize
    
    if len(embeddings) == 0:
        return {}
    
    # Normalize embeddings
    embeddings_norm = normalize(embeddings)
    
    # DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings_norm)
    
    # Create mapping
    cluster_map = {item_id: int(label) for item_id, label in zip(item_ids, labels)}
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"✓ Face clustering: {n_clusters} clusters, {n_noise} noise points")
    
    return cluster_map


# ============================================================
# 10. DUPLICATE DETECTION (pHash)
# ============================================================

def detect_duplicates(phashes: Dict[str, str], threshold: int = 5) -> Dict[str, Tuple[int, bool]]:
    """
    Detect duplicates using pHash hamming distance.
    Returns mapping of item_id -> (duplicate_group_id, is_duplicate)
    """
    import imagehash
    
    items = list(phashes.items())
    n = len(items)
    
    # Union-Find for grouping
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Compare all pairs (O(n²) but ok for ~2500 images)
    for i in range(n):
        if not items[i][1]:
            continue
        hash_i = imagehash.hex_to_hash(items[i][1])
        
        for j in range(i + 1, n):
            if not items[j][1]:
                continue
            hash_j = imagehash.hex_to_hash(items[j][1])
            
            # Hamming distance
            distance = hash_i - hash_j
            if distance <= threshold:
                union(i, j)
    
    # Build result
    result = {}
    group_ids = {}
    next_group = 0
    
    for i, (item_id, _) in enumerate(items):
        root = find(i)
        
        if root not in group_ids:
            group_ids[root] = next_group
            next_group += 1
        
        group_id = group_ids[root]
        
        # Check if this item has duplicates (group size > 1)
        group_members = [j for j in range(n) if find(j) == root]
        is_duplicate = len(group_members) > 1
        
        result[item_id] = (group_id, is_duplicate)
    
    n_duplicates = sum(1 for _, is_dup in result.values() if is_dup)
    print(f"✓ Duplicate detection: {n_duplicates} duplicates found")
    
    return result


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_feature_extraction(
    sample_size: int = None,
    skip_existing: bool = True
):
    """
    Main pipeline to extract all image features.
    
    Output: Parquet file with all features for ML training.
    """
    print("\n" + "=" * 60)
    print("IMAGE FEATURE EXTRACTION PIPELINE")
    print("=" * 60)
    
    setup_directories()
    
    # Get all images
    image_files = list(DIRS['images'].glob("*.jpg")) + list(DIRS['images'].glob("*.png"))
    
    if sample_size:
        image_files = image_files[:sample_size]
    
    print(f"\n📷 Processing {len(image_files)} images")
    
    # Initialize extractors
    face_detector = FaceDetector()
    clip_zeroshot = CLIPZeroShot()
    
    # Collect all features
    all_features = []
    all_phashes = {}
    face_embeddings = []
    face_item_ids = []
    
    # ========== PHASE 1: Per-image extraction ==========
    print("\n" + "-" * 40)
    print("PHASE 1: Extracting per-image features")
    print("-" * 40)
    
    for img_path in tqdm(image_files, desc="Extracting"):
        item_id = img_path.stem
        
        features = {'item_id': item_id, 'image_path': str(img_path)}
        
        # Step 2: Metadata
        meta = extract_metadata(img_path)
        features.update(meta)
        
        # Step 3: Blur score
        blur_score, is_blurry = calculate_blur_score(img_path)
        features['blur_score'] = blur_score
        features['is_blurry'] = is_blurry
        
        # Step 4: pHash
        phash = calculate_phash(img_path)
        features['phash'] = phash
        all_phashes[item_id] = phash
        
        # Step 5: Face detection
        face_info = face_detector.detect(str(img_path))
        features.update(face_info)
        
        # Step 6-7: Load existing embeddings
        clip_emb = load_existing_embedding(item_id, 'clip')
        face_emb = load_existing_embedding(item_id, 'face')
        
        features['has_clip_embedding'] = clip_emb is not None
        features['has_face_embedding'] = face_emb is not None
        
        # Collect face embeddings for clustering
        if face_emb is not None:
            face_embeddings.append(face_emb)
            face_item_ids.append(item_id)
        
        # Step 8: CLIP zero-shot
        zeroshot = clip_zeroshot.classify(str(img_path))
        features['top_label'] = zeroshot['top_label']
        features['top_label_score'] = zeroshot['top_score']
        features['label_scores'] = json.dumps(zeroshot['label_scores'])
        
        features['status'] = 'done'
        features['created_at'] = datetime.now().isoformat()
        
        all_features.append(features)
    
    # Clear GPU cache
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()
    
    # ========== PHASE 2: Face clustering ==========
    print("\n" + "-" * 40)
    print("PHASE 2: Face clustering (DBSCAN)")
    print("-" * 40)
    
    if face_embeddings:
        face_emb_array = np.array(face_embeddings)
        cluster_map = cluster_faces(face_emb_array, face_item_ids)
        
        # Add cluster IDs to features
        for features in all_features:
            features['face_cluster_id'] = cluster_map.get(features['item_id'], -1)
    else:
        for features in all_features:
            features['face_cluster_id'] = -1
    
    # ========== PHASE 3: Duplicate detection ==========
    print("\n" + "-" * 40)
    print("PHASE 3: Duplicate detection (pHash)")
    print("-" * 40)
    
    duplicate_map = detect_duplicates(all_phashes)
    
    for features in all_features:
        dup_info = duplicate_map.get(features['item_id'], (0, False))
        features['duplicate_group_id'] = dup_info[0]
        features['is_duplicate'] = dup_info[1]
    
    # ========== SAVE RESULTS ==========
    print("\n" + "-" * 40)
    print("Saving results...")
    print("-" * 40)
    
    df = pd.DataFrame(all_features)
    
    # Save as Parquet (best for ML)
    df.to_parquet(FILES['output_parquet'], index=False)
    print(f"✓ Saved: {FILES['output_parquet']}")
    
    # Save as CSV (for easy viewing)
    df.to_csv(FILES['output_csv'], index=False)
    print(f"✓ Saved: {FILES['output_csv']}")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    print(f"\nTotal images: {len(df)}")
    print(f"With faces: {(df['num_faces'] > 0).sum()}")
    print(f"Blurry images: {df['is_blurry'].sum()}")
    print(f"Duplicates: {df['is_duplicate'].sum()}")
    print(f"Face clusters: {df['face_cluster_id'].nunique()}")
    
    print(f"\nTop labels distribution:")
    print(df['top_label'].value_counts().head(5))
    
    print(f"\nOutput files:")
    print(f"  - {FILES['output_parquet']}")
    print(f"  - {FILES['output_csv']}")
    
    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Feature Extraction")
    parser.add_argument("--sample", type=int, help="Process only N images")
    
    args = parser.parse_args()
    
    df = run_feature_extraction(sample_size=args.sample)
