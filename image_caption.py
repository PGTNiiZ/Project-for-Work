"""
Image Captioning Pipeline with BLIP-2 + Multilingual Text Embeddings
=====================================================================
This script:
1. Generates captions from images using BLIP-2 (opt-2.7b)
2. Creates multilingual text embeddings from captions
3. Stores results in SQLite for bio/post similarity matching

Pipeline:
    Image → BLIP Caption (EN) → Multilingual Text Embedding
                                        ↓
                            caption ↔ bio/post similarity
"""

import os
import sys
import sqlite3
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GPU CONFIGURATION
# ============================================================

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

if CUDA_AVAILABLE:
    print(f"🚀 CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ No CUDA GPU detected, running on CPU")

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"

# Directory structure
DIRS = {
    'images': DATA_DIR / "images",
    'captions': DATA_DIR / "output" / "captions",
    'caption_embeddings': DATA_DIR / "output" / "captions" / "embeddings",
    'manifest': DATA_DIR / "output" / "manifest",
}

# Files
FILES = {
    'caption_db': DIRS['manifest'] / "caption_progress.sqlite",
}

# Model settings
# BLIP-1 base model (~1GB) - faster and smaller than BLIP-2 (~15GB)
# Options: "Salesforce/blip-image-captioning-base" (faster), "Salesforce/blip-image-captioning-large" (better)
BLIP_MODEL = "Salesforce/blip-image-captioning-large"
TEXT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_EMBED_DIM = 384

# Processing settings
BATCH_SIZE = 16 if CUDA_AVAILABLE else 4  # BLIP-1 uses less VRAM


# ============================================================
# 1. DIRECTORY SETUP
# ============================================================

def setup_directories():
    """Create required directories."""
    for name, path in DIRS.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {path}")
    print("\n✅ Directory structure ready!")


# ============================================================
# 2. CAPTION DATABASE (SQLite)
# ============================================================

class CaptionDatabase:
    """SQLite database for captions and embeddings."""
    
    def __init__(self, db_path: Path = FILES['caption_db']):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._create_tables()
    
    def _create_tables(self):
        """Create captions and caption_embeddings tables."""
        cursor = self.conn.cursor()
        
        # Main captions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS captions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT NOT NULL,
                url TEXT,
                image_path TEXT,
                image_hash TEXT,
                caption_en TEXT,
                caption_th TEXT,
                model_name TEXT DEFAULT 'blip2-opt-2.7b',
                status TEXT DEFAULT 'pending',
                error_msg TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(item_id, image_hash)
            )
        ''')
        
        # Caption embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS caption_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT NOT NULL,
                caption_text TEXT,
                embed_model TEXT,
                embedding BLOB,
                dim INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_captions_item ON captions(item_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_captions_status ON captions(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_item ON caption_embeddings(item_id)')
        
        self.conn.commit()
        print("✓ Caption database ready")
    
    def add_caption(self, item_id: str, image_path: str, caption: str,
                   url: str = None, image_hash: str = None) -> bool:
        """Add or update a caption."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO captions 
                (item_id, url, image_path, image_hash, caption_en, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'done', CURRENT_TIMESTAMP)
            ''', (item_id, url, str(image_path), image_hash, caption))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"DB error: {e}")
            return False
    
    def mark_failed(self, item_id: str, image_path: str, error_msg: str):
        """Mark a caption as failed."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO captions 
            (item_id, image_path, status, error_msg, created_at)
            VALUES (?, ?, 'failed', ?, CURRENT_TIMESTAMP)
        ''', (item_id, str(image_path), error_msg[:500]))
        self.conn.commit()
    
    def add_embedding(self, item_id: str, caption_text: str, 
                     embedding: np.ndarray, model_name: str):
        """Add caption embedding."""
        cursor = self.conn.cursor()
        emb_bytes = embedding.astype(np.float16).tobytes()
        cursor.execute('''
            INSERT INTO caption_embeddings 
            (item_id, caption_text, embed_model, embedding, dim)
            VALUES (?, ?, ?, ?, ?)
        ''', (item_id, caption_text, model_name, emb_bytes, len(embedding)))
        self.conn.commit()
    
    def get_pending_items(self, limit: int = None) -> List[str]:
        """Get item_ids that haven't been processed."""
        cursor = self.conn.cursor()
        query = "SELECT DISTINCT item_id FROM captions WHERE status != 'done'"
        if limit:
            query += f" LIMIT {limit}"
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]
    
    def get_items_needing_embedding(self) -> List[Tuple[str, str]]:
        """Get captions that need embeddings."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT c.item_id, c.caption_en 
            FROM captions c
            LEFT JOIN caption_embeddings e ON c.item_id = e.item_id
            WHERE c.status = 'done' AND c.caption_en IS NOT NULL AND e.id IS NULL
        ''')
        return cursor.fetchall()
    
    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Get all embeddings for similarity search."""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT item_id, embedding, dim FROM caption_embeddings
        ''')
        rows = cursor.fetchall()
        
        if not rows:
            return np.array([]), []
        
        item_ids = [row[0] for row in rows]
        embeddings = [np.frombuffer(row[1], dtype=np.float16).astype(np.float32) for row in rows]
        
        return np.array(embeddings), item_ids
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        cursor = self.conn.cursor()
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM captions WHERE status = 'done'")
        stats['captions_done'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM captions WHERE status = 'failed'")
        stats['captions_failed'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM caption_embeddings")
        stats['embeddings'] = cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        self.conn.close()


# ============================================================
# 3. BLIP CAPTIONER (GPU + FP16) - Using BLIP-1 (smaller model)
# ============================================================

class BLIPCaptioner:
    """Generate image captions using BLIP with GPU acceleration."""
    
    def __init__(self, model_name: str = BLIP_MODEL):
        print(f"Loading BLIP model: {model_name}...")
        print("  This may take a minute on first run...")
        
        self.device = DEVICE
        self.model = None
        self.processor = None
        self.available = False
        
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            # Load processor
            self.processor = BlipProcessor.from_pretrained(model_name)
            
            # Load model with optimizations
            if CUDA_AVAILABLE:
                # Use FP16 for GPU
                self.model = BlipForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16
                ).to(self.device)
                print(f"✓ BLIP loaded on GPU with FP16")
            else:
                self.model = BlipForConditionalGeneration.from_pretrained(model_name)
                self.model.to(self.device)
                print(f"✓ BLIP loaded on CPU")
            
            self.model.eval()
            self.available = True
            
        except Exception as e:
            print(f"✗ BLIP load error: {e}")
            print("  Install: pip install transformers")
            self.available = False
    
    def generate_caption(self, image_path: str, max_length: int = 50) -> Optional[str]:
        """Generate single caption."""
        if not self.available:
            return None
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Resize large images to save VRAM
            max_size = 384
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Process with conditional captioning (empty text prompt)
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            if CUDA_AVAILABLE:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    num_beams=3,
                    early_stopping=True
                )
            
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            return caption.strip()
            
        except Exception as e:
            return None
    
    def generate_captions_batch(self, image_paths: List[str], 
                               max_length: int = 50) -> List[Optional[str]]:
        """Generate captions for batch of images."""
        if not self.available:
            return [None] * len(image_paths)
        
        captions = []
        for path in image_paths:
            caption = self.generate_caption(path, max_length)
            captions.append(caption)
        
        # Clear GPU cache after batch
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
        
        return captions


# ============================================================
# 4. MULTILINGUAL TEXT EMBEDDER (GPU)
# ============================================================

class MultilingualTextEmbedder:
    """Generate multilingual text embeddings with GPU support."""
    
    def __init__(self, model_name: str = TEXT_EMBED_MODEL):
        print(f"Loading Text Embedding model: {model_name}...")
        
        self.device = DEVICE
        self.model = None
        self.available = False
        self.dim = TEXT_EMBED_DIM
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Enable FP16 on GPU
            if CUDA_AVAILABLE:
                self.model = self.model.half()
                print(f"✓ Text Embedder loaded on GPU with FP16")
            else:
                print(f"✓ Text Embedder loaded on CPU")
            
            self.available = True
            
        except Exception as e:
            print(f"✗ Text Embedder error: {e}")
            print("  Install: pip install sentence-transformers")
            self.available = False
    
    def encode(self, text: str) -> Optional[np.ndarray]:
        """Encode single text."""
        if not self.available or not text:
            return None
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding / np.linalg.norm(embedding)
        except:
            return None
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> List[Optional[np.ndarray]]:
        """Encode batch of texts."""
        if not self.available:
            return [None] * len(texts)
        
        results = []
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text:
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            return [None] * len(texts)
        
        try:
            embeddings = self.model.encode(
                valid_texts, 
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Normalize
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
            # Map back
            result = [None] * len(texts)
            for idx, emb in zip(valid_indices, embeddings):
                result[idx] = emb
            
            return result
            
        except Exception as e:
            print(f"Batch encode error: {e}")
            return [None] * len(texts)


# ============================================================
# 5. MAIN CAPTION PIPELINE
# ============================================================

def run_caption_pipeline(
    input_csv: Path = None,
    images_dir: Path = None,
    sample_size: int = None,
    skip_caption: bool = False,
    skip_embedding: bool = False
):
    """
    Main pipeline to generate captions and embeddings.
    
    Args:
        input_csv: CSV with profile data (uses existing images)
        images_dir: Directory containing images to caption
        sample_size: Limit number of images
        skip_caption: Skip caption generation (only embeddings)
        skip_embedding: Skip embedding generation (only captions)
    """
    print("\n" + "=" * 60)
    print("IMAGE CAPTION PIPELINE (BLIP-2 + Multilingual Embeddings)")
    print("=" * 60)
    
    # Setup
    setup_directories()
    db = CaptionDatabase()
    
    # Find images
    if images_dir is None:
        images_dir = DIRS['images']
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if sample_size:
        image_files = image_files[:sample_size]
    
    print(f"\n📷 Found {len(image_files)} images")
    
    # ========== PHASE 1: Generate Captions ==========
    if not skip_caption:
        print("\n" + "-" * 40)
        print("PHASE 1: Generating Captions (BLIP-2)")
        print("-" * 40)
        
        captioner = BLIPCaptioner()
        
        if captioner.available:
            success = 0
            failed = 0
            
            for img_path in tqdm(image_files, desc="Captioning"):
                item_id = img_path.stem  # filename without extension
                
                try:
                    caption = captioner.generate_caption(str(img_path))
                    
                    if caption:
                        # Calculate image hash for dedup
                        with open(img_path, 'rb') as f:
                            img_hash = hashlib.md5(f.read()).hexdigest()[:16]
                        
                        db.add_caption(item_id, str(img_path), caption, image_hash=img_hash)
                        success += 1
                    else:
                        db.mark_failed(item_id, str(img_path), "No caption generated")
                        failed += 1
                        
                except Exception as e:
                    db.mark_failed(item_id, str(img_path), str(e))
                    failed += 1
            
            print(f"\n✓ Captions: {success} success, {failed} failed")
        else:
            print("⚠️ BLIP model not available, skipping captions")
    
    # ========== PHASE 2: Generate Embeddings ==========
    if not skip_embedding:
        print("\n" + "-" * 40)
        print("PHASE 2: Generating Text Embeddings")
        print("-" * 40)
        
        embedder = MultilingualTextEmbedder()
        
        if embedder.available:
            # Get captions needing embeddings
            items = db.get_items_needing_embedding()
            print(f"   {len(items)} captions need embeddings")
            
            if items:
                for item_id, caption in tqdm(items, desc="Embedding"):
                    embedding = embedder.encode(caption)
                    
                    if embedding is not None:
                        db.add_embedding(item_id, caption, embedding, TEXT_EMBED_MODEL)
            
            print(f"✓ Embeddings generated")
        else:
            print("⚠️ Text Embedder not available, skipping embeddings")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 60)
    print("CAPTION PIPELINE COMPLETE!")
    print("=" * 60)
    
    stats = db.get_stats()
    print(f"Captions done: {stats['captions_done']}")
    print(f"Captions failed: {stats['captions_failed']}")
    print(f"Text embeddings: {stats['embeddings']}")
    print(f"\nDatabase: {FILES['caption_db']}")
    
    db.close()
    return stats


# ============================================================
# 6. SIMILARITY SEARCH UTILITY
# ============================================================

def search_similar_captions(query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Find profiles with similar captions to query text.
    Can be used to match bio/post text with image captions.
    
    Args:
        query_text: Bio or post text to match
        top_k: Number of results
        
    Returns:
        List of (item_id, similarity_score) tuples
    """
    embedder = MultilingualTextEmbedder()
    if not embedder.available:
        return []
    
    db = CaptionDatabase()
    
    # Get query embedding
    query_emb = embedder.encode(query_text)
    if query_emb is None:
        return []
    
    # Get all caption embeddings
    embeddings, item_ids = db.get_all_embeddings()
    
    if len(embeddings) == 0:
        return []
    
    # Compute cosine similarities
    similarities = np.dot(embeddings, query_emb)
    
    # Get top-k
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = [(item_ids[i], float(similarities[i])) for i in top_indices]
    
    db.close()
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Caption Pipeline")
    parser.add_argument("--sample", type=int, default=None, 
                       help="Process only N images (for testing)")
    parser.add_argument("--skip-caption", action="store_true",
                       help="Skip caption generation")
    parser.add_argument("--skip-embedding", action="store_true",
                       help="Skip embedding generation")
    parser.add_argument("--images-dir", type=str, default=None,
                       help="Custom images directory")
    
    args = parser.parse_args()
    
    images_dir = Path(args.images_dir) if args.images_dir else None
    
    run_caption_pipeline(
        images_dir=images_dir,
        sample_size=args.sample,
        skip_caption=args.skip_caption,
        skip_embedding=args.skip_embedding
    )
