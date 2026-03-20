"""
Advanced Embedding Pipeline
============================
Features:
- ArcFace embeddings for face recognition
- CLIP embeddings for general image understanding
- FAISS index for fast similarity search
- SQLite progress tracking (resumable)
- URL status logging to Parquet

Directory Structure:
├─ embeddings/
│  ├─ face/           # ArcFace embeddings
│  ├─ clip/           # CLIP embeddings
│  └─ fused/          # Late fusion vectors
├─ index/
│  ├─ faiss_face.index
│  └─ faiss_clip.index
├─ logs/
└─ manifest/
   ├─ progress.sqlite
   └─ url_status.parquet
"""

import os
import sys
import sqlite3
import hashlib
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GPU CONFIGURATION
# ============================================================

# Check GPU availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    # Set TensorFlow to use GPU
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
    # Enable GPU memory growth for TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   TensorFlow GPUs: {len(gpus)} detected")
    except:
        pass
else:
    print("⚠️ No CUDA GPU detected, running on CPU")

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Directory structure
DIRS = {
    'images': DATA_DIR / "images",
    'embeddings_face': DATA_DIR / "embeddings" / "face",
    'embeddings_clip': DATA_DIR / "embeddings" / "clip",
    'embeddings_fused': DATA_DIR / "embeddings" / "fused",
    'index': DATA_DIR / "index",
    'logs': DATA_DIR / "logs",
    'manifest': DATA_DIR / "manifest",
    'processed': DATA_DIR / "processed",
}

# Files
FILES = {
    'progress_db': DIRS['manifest'] / "progress.sqlite",
    'url_status': DIRS['manifest'] / "url_status.parquet",
    'faiss_face': DIRS['index'] / "faiss_face.index",
    'faiss_clip': DIRS['index'] / "faiss_clip.index",
}

# Settings
TIMEOUT = 15
BATCH_SIZE = 64 if CUDA_AVAILABLE else 16  # Larger batch on GPU
CLIP_DIM = 512
ARCFACE_DIM = 512
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# ============================================================
# 1. SETUP & DIRECTORY CREATION
# ============================================================

def setup_directories():
    """Create all required directories."""
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {path}")
    
    print("\n✅ Directory structure ready!")


def setup_logging():
    """Setup logging to file and console."""
    log_file = DIRS['logs'] / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ============================================================
# 2. PROGRESS TRACKER (SQLite)
# ============================================================

class ProgressTracker:
    """Track processing progress with SQLite for resumable pipeline."""
    
    def __init__(self, db_path: Path = FILES['progress_db']):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self._create_tables()
    
    def _create_tables(self):
        """Create progress tables."""
        cursor = self.conn.cursor()
        
        # Main progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS progress (
                profile_id TEXT PRIMARY KEY,
                platform TEXT,
                image_status TEXT DEFAULT 'pending',
                face_embedding_status TEXT DEFAULT 'pending',
                clip_embedding_status TEXT DEFAULT 'pending',
                image_path TEXT,
                face_embedding_path TEXT,
                clip_embedding_path TEXT,
                error_message TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # URL status table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS url_status (
                url_hash TEXT PRIMARY KEY,
                url TEXT,
                http_status INTEGER,
                content_type TEXT,
                is_valid BOOLEAN,
                error_message TEXT,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def get_pending(self, task: str = 'image') -> List[str]:
        """Get list of pending profile_ids for a specific task."""
        status_col = f"{task}_status" if task != 'image' else 'image_status'
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT profile_id FROM progress WHERE {status_col} = 'pending'")
        return [row[0] for row in cursor.fetchall()]
    
    def update_status(self, profile_id: str, **kwargs):
        """Update status for a profile."""
        cursor = self.conn.cursor()
        
        # Check if exists
        cursor.execute("SELECT 1 FROM progress WHERE profile_id = ?", (profile_id,))
        exists = cursor.fetchone()
        
        kwargs['updated_at'] = datetime.now()
        
        if exists:
            set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
            values = list(kwargs.values()) + [profile_id]
            cursor.execute(f"UPDATE progress SET {set_clause} WHERE profile_id = ?", values)
        else:
            kwargs['profile_id'] = profile_id
            cols = ", ".join(kwargs.keys())
            placeholders = ", ".join(["?" for _ in kwargs])
            cursor.execute(f"INSERT INTO progress ({cols}) VALUES ({placeholders})", list(kwargs.values()))
        
        self.conn.commit()
    
    def log_url_status(self, url: str, http_status: int, content_type: str, 
                       is_valid: bool, error_message: str = None):
        """Log URL check result."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO url_status 
            (url_hash, url, http_status, content_type, is_valid, error_message, checked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (url_hash, url, http_status, content_type, is_valid, error_message, datetime.now()))
        
        self.conn.commit()
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        for status in ['pending', 'done', 'failed']:
            cursor.execute("SELECT COUNT(*) FROM progress WHERE image_status = ?", (status,))
            stats[f'image_{status}'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM progress WHERE face_embedding_status = ?", (status,))
            stats[f'face_{status}'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM progress WHERE clip_embedding_status = ?", (status,))
            stats[f'clip_{status}'] = cursor.fetchone()[0]
        
        return stats
    
    def export_url_status_parquet(self):
        """Export URL status to Parquet file."""
        df = pd.read_sql("SELECT * FROM url_status", self.conn)
        df.to_parquet(FILES['url_status'], index=False)
        print(f"✓ Exported URL status to {FILES['url_status']}")
    
    def close(self):
        self.conn.close()


# ============================================================
# 3. IMAGE DOWNLOADER (with progress tracking)
# ============================================================

def download_image(url: str, save_path: Path, tracker: ProgressTracker) -> Tuple[bool, str]:
    """Download image and log status."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=TIMEOUT, stream=True)
        
        http_status = response.status_code
        content_type = response.headers.get('content-type', '')
        
        if http_status != 200:
            tracker.log_url_status(url, http_status, content_type, False, f"HTTP {http_status}")
            return False, f"HTTP {http_status}"
        
        if 'image' not in content_type.lower():
            tracker.log_url_status(url, http_status, content_type, False, "Not an image")
            return False, "Not an image"
        
        # Validate and save image
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img.save(str(save_path), 'JPEG', quality=85)
        tracker.log_url_status(url, http_status, content_type, True)
        return True, "OK"
        
    except Exception as e:
        tracker.log_url_status(url, 0, "", False, str(e)[:100])
        return False, str(e)[:100]


# ============================================================
# 4. CLIP EMBEDDER
# ============================================================

class CLIPEmbedder:
    """Extract CLIP embeddings with GPU support."""
    
    def __init__(self):
        print(f"Loading CLIP model on {DEVICE}...")
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = DEVICE
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device).eval()
        
        # Enable half precision on GPU for faster inference
        if self.device == "cuda":
            self.model = self.model.half()
            print(f"✓ CLIP loaded on GPU with FP16 (faster)")
        else:
            print(f"✓ CLIP loaded on {self.device}")
    
    def extract(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use half precision on GPU
            if self.device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            
            emb = features.float().cpu().numpy().flatten()
            return emb / np.linalg.norm(emb)
        except Exception as e:
            return None
    
    def extract_batch(self, image_paths: List[str], batch_size: int = BATCH_SIZE) -> List[Optional[np.ndarray]]:
        """Extract embeddings in batches for GPU efficiency."""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            for j, path in enumerate(batch_paths):
                if path and Path(path).exists():
                    try:
                        img = Image.open(path).convert('RGB')
                        batch_images.append(img)
                        valid_indices.append(j)
                    except:
                        pass
            
            if not batch_images:
                results.extend([None] * len(batch_paths))
                continue
            
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                if self.device == "cuda":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                
                batch_embeddings = features.float().cpu().numpy()
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                
                # Map back to original order
                result = [None] * len(batch_paths)
                for idx, emb in zip(valid_indices, batch_embeddings):
                    result[idx] = emb
                
                results.extend(result)
            except Exception as e:
                print(f"Batch error: {e}")
                results.extend([None] * len(batch_paths))
        
        return results


# ============================================================
# 5. FACE EMBEDDER (using DeepFace with ArcFace model)
# ============================================================

class FaceEmbedder:
    """Extract face embeddings using DeepFace with ArcFace model."""
    
    def __init__(self, model_name: str = "ArcFace"):
        """
        Initialize face embedding model.
        
        Args:
            model_name: DeepFace model to use. Options:
                - "ArcFace" (recommended, 512-dim)
                - "Facenet" (128-dim)
                - "VGG-Face" (2622-dim)
                - "Facenet512" (512-dim)
        """
        print(f"Loading Face model ({model_name})...")
        self.model_name = model_name
        
        try:
            from deepface import DeepFace
            self.DeepFace = DeepFace
            
            # Pre-load model by running on dummy
            print("  Warming up model (first run downloads weights)...")
            self.available = True
            print(f"✓ DeepFace ({model_name}) ready")
        except ImportError:
            print("⚠ DeepFace not installed. Run: pip install deepface tf-keras")
            self.available = False
        except Exception as e:
            print(f"⚠ Face model load error: {e}")
            self.available = False
    
    def extract(self, image_path: str) -> Optional[np.ndarray]:
        """Extract face embedding from image."""
        if not self.available:
            return None
        
        try:
            # DeepFace returns embedding as list
            # Use retinaface for better GPU detection if available
            detector = "retinaface" if CUDA_AVAILABLE else "opencv"
            result = self.DeepFace.represent(
                img_path=str(image_path),
                model_name=self.model_name,
                enforce_detection=False,  # Don't fail if face not detected
                detector_backend=detector
            )
            
            if not result:
                return None
            
            # Get first face's embedding
            emb = np.array(result[0]["embedding"])
            return emb / np.linalg.norm(emb)
            
        except Exception as e:
            return None


# Alias for backward compatibility
ArcFaceEmbedder = FaceEmbedder


# ============================================================
# 6. FAISS INDEX BUILDER
# ============================================================

class FAISSIndexBuilder:
    """Build and manage FAISS indexes for similarity search with GPU support."""
    
    def __init__(self):
        self.use_gpu = False
        try:
            import faiss
            self.faiss = faiss
            self.available = True
            
            # Check for GPU FAISS
            if CUDA_AVAILABLE:
                try:
                    # Check if faiss-gpu is installed
                    res = faiss.StandardGpuResources()
                    self.gpu_resources = res
                    self.use_gpu = True
                    print("✓ FAISS-GPU available (accelerated)")
                except AttributeError:
                    print("✓ FAISS-CPU available (install faiss-gpu for GPU acceleration)")
            else:
                print("✓ FAISS available (CPU)")
        except ImportError:
            print("⚠ FAISS not installed. Run: pip install faiss-cpu")
            self.available = False
    
    def build_index(self, embeddings: np.ndarray, index_path: Path, 
                    use_gpu: bool = None) -> bool:
        """Build FAISS index from embeddings."""
        if not self.available:
            return False
        
        # Use GPU by default if available
        if use_gpu is None:
            use_gpu = self.use_gpu
        
        try:
            dim = embeddings.shape[1]
            
            # Use IVF index for large datasets
            if len(embeddings) > 10000:
                nlist = min(int(np.sqrt(len(embeddings))), 1024)
                quantizer = self.faiss.IndexFlatIP(dim)
                index = self.faiss.IndexIVFFlat(quantizer, dim, nlist, self.faiss.METRIC_INNER_PRODUCT)
                
                # Train on GPU if available
                if use_gpu and self.use_gpu:
                    gpu_index = self.faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
                    gpu_index.train(embeddings)
                    gpu_index.add(embeddings)
                    # Convert back to CPU for saving
                    index = self.faiss.index_gpu_to_cpu(gpu_index)
                else:
                    index.train(embeddings)
                    index.add(embeddings)
            else:
                index = self.faiss.IndexFlatIP(dim)
                index.add(embeddings)
            
            self.faiss.write_index(index, str(index_path))
            gpu_str = " (built on GPU)" if (use_gpu and self.use_gpu) else ""
            print(f"✓ FAISS index saved: {index_path} ({len(embeddings)} vectors){gpu_str}")
            return True
        except Exception as e:
            print(f"✗ FAISS index error: {e}")
            return False
    
    def load_index(self, index_path: Path, to_gpu: bool = None):
        """Load existing FAISS index, optionally to GPU."""
        if not self.available or not index_path.exists():
            return None
        
        index = self.faiss.read_index(str(index_path))
        
        # Move to GPU if requested and available
        if to_gpu is None:
            to_gpu = self.use_gpu
        if to_gpu and self.use_gpu:
            index = self.faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        
        return index
    
    def search(self, index, query: np.ndarray, k: int = 10):
        """Search similar vectors."""
        if index is None:
            return None, None
        
        query = query.reshape(1, -1).astype('float32')
        scores, indices = index.search(query, k)
        return scores[0], indices[0]


# ============================================================
# 7. EMBEDDING FUSION
# ============================================================

def fuse_embeddings(face_emb: Optional[np.ndarray], 
                    clip_emb: Optional[np.ndarray],
                    method: str = 'concat') -> Optional[np.ndarray]:
    """
    Fuse face and CLIP embeddings.
    
    Methods:
    - concat: Simple concatenation [face; clip]
    - average: Element-wise average (requires same dim)
    - weighted: Weighted combination
    """
    if face_emb is None and clip_emb is None:
        return None
    
    if method == 'concat':
        if face_emb is None:
            face_emb = np.zeros(ARCFACE_DIM)
        if clip_emb is None:
            clip_emb = np.zeros(CLIP_DIM)
        fused = np.concatenate([face_emb, clip_emb])
    
    elif method == 'weighted':
        # Weight face more for identity matching
        alpha = 0.7  # face weight
        if face_emb is not None and clip_emb is not None:
            # Pad to same dimension if needed
            if len(face_emb) != len(clip_emb):
                max_dim = max(len(face_emb), len(clip_emb))
                face_emb = np.pad(face_emb, (0, max_dim - len(face_emb)))
                clip_emb = np.pad(clip_emb, (0, max_dim - len(clip_emb)))
            fused = alpha * face_emb + (1 - alpha) * clip_emb
        else:
            fused = face_emb if face_emb is not None else clip_emb
    
    else:  # average
        if face_emb is not None and clip_emb is not None:
            fused = (face_emb + clip_emb) / 2
        else:
            fused = face_emb if face_emb is not None else clip_emb
    
    # Normalize
    if fused is not None:
        fused = fused / np.linalg.norm(fused)
    
    return fused


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def run_advanced_pipeline(
    input_csv: Path = None,
    skip_download: bool = False,
    skip_face: bool = False,
    sample_size: int = None
):
    """
    Run the complete advanced embedding pipeline.
    
    Args:
        input_csv: Input CSV with profile data
        skip_download: Skip image download (use existing)
        skip_face: Skip ArcFace (if not installed)
        sample_size: Limit processing for testing
    """
    print("=" * 60)
    print("ADVANCED EMBEDDING PIPELINE")
    print("=" * 60)
    
    # Setup
    setup_directories()
    logger = setup_logging()
    tracker = ProgressTracker()
    
    # Default input
    if input_csv is None:
        input_csv = DIRS['processed'] / "all_profiles_cleaned.csv"
    
    # Load data
    print(f"\n1. Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    print(f"   Loaded {len(df)} profiles")
    
    # Initialize progress for all profiles
    for _, row in df.iterrows():
        tracker.update_status(
            row['profile_id'],
            platform=row.get('platform', 'unknown')
        )
    
    # Initialize models
    print("\n2. Loading models...")
    clip_model = CLIPEmbedder()
    
    if not skip_face:
        face_model = FaceEmbedder(model_name="ArcFace")
    else:
        face_model = None
    
    faiss_builder = FAISSIndexBuilder()
    
    # Process profiles
    print(f"\n3. Processing {len(df)} profiles...")
    
    clip_embeddings = []
    face_embeddings = []
    profile_ids = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        profile_id = row['profile_id']
        platform = row.get('platform', 'unknown')
        url = row.get('pictureURL', '')
        
        # Image path
        url_hash = hashlib.md5(str(url).encode()).hexdigest()[:8]
        img_path = DIRS['images'] / f"{profile_id}_{platform}_{url_hash}.jpg"
        
        # Download image if needed
        if not skip_download and not img_path.exists():
            if pd.notna(url) and str(url).startswith('http'):
                success, msg = download_image(url, img_path, tracker)
                if success:
                    tracker.update_status(profile_id, image_status='done', image_path=str(img_path))
                else:
                    tracker.update_status(profile_id, image_status='failed', error_message=msg)
                time.sleep(0.05)  # Rate limit
        
        # Skip if no image
        if not img_path.exists():
            continue
        
        # CLIP embedding
        clip_path = DIRS['embeddings_clip'] / f"{profile_id}_{platform}.npy"
        if not clip_path.exists():
            clip_emb = clip_model.extract(str(img_path))
            if clip_emb is not None:
                np.save(clip_path, clip_emb)
                tracker.update_status(profile_id, clip_embedding_status='done', 
                                     clip_embedding_path=str(clip_path))
                clip_embeddings.append(clip_emb)
                profile_ids.append(profile_id)
        else:
            clip_emb = np.load(clip_path)
            clip_embeddings.append(clip_emb)
            profile_ids.append(profile_id)
        
        # ArcFace embedding
        if face_model and face_model.available:
            face_path = DIRS['embeddings_face'] / f"{profile_id}_{platform}.npy"
            if not face_path.exists():
                face_emb = face_model.extract(str(img_path))
                if face_emb is not None:
                    np.save(face_path, face_emb)
                    tracker.update_status(profile_id, face_embedding_status='done',
                                         face_embedding_path=str(face_path))
                    face_embeddings.append(face_emb)
            else:
                face_emb = np.load(face_path)
                face_embeddings.append(face_emb)
    
    # Build FAISS indexes
    print("\n4. Building FAISS indexes...")
    
    if clip_embeddings:
        clip_matrix = np.vstack(clip_embeddings).astype('float32')
        faiss_builder.build_index(clip_matrix, FILES['faiss_clip'])
        
        # Save ID mapping
        id_map = pd.DataFrame({'idx': range(len(profile_ids)), 'profile_id': profile_ids})
        id_map.to_parquet(DIRS['index'] / "clip_id_map.parquet", index=False)
    
    if face_embeddings:
        face_matrix = np.vstack(face_embeddings).astype('float32')
        faiss_builder.build_index(face_matrix, FILES['faiss_face'])
    
    # Export URL status
    print("\n5. Exporting manifest files...")
    tracker.export_url_status_parquet()
    
    # Summary
    stats = tracker.get_stats()
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Images: {stats['image_done']} done, {stats['image_failed']} failed")
    print(f"CLIP embeddings: {len(clip_embeddings)}")
    print(f"Face embeddings: {len(face_embeddings)}")
    print(f"\nOutputs:")
    print(f"  - CLIP index: {FILES['faiss_clip']}")
    print(f"  - Face index: {FILES['faiss_face']}")
    print(f"  - Progress DB: {FILES['progress_db']}")
    print(f"  - URL status: {FILES['url_status']}")
    
    tracker.close()
    return stats


# ============================================================
# USAGE
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Embedding Pipeline")
    parser.add_argument('--sample', type=int, default=None, help="Sample size for testing")
    parser.add_argument('--skip-download', action='store_true', help="Skip image download")
    parser.add_argument('--skip-face', action='store_true', help="Skip ArcFace embeddings")
    
    args = parser.parse_args()
    
    run_advanced_pipeline(
        sample_size=args.sample,
        skip_download=args.skip_download,
        skip_face=args.skip_face
    )
