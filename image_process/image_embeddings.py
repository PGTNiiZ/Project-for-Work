"""
Image Downloader and Embedding Extractor
=========================================
This script:
1. Downloads profile images from pictureURL (skips dead links)
2. Extracts image embeddings using CLIP (OpenAI) pre-trained model

CLIP is chosen because:
- Works well with diverse image types (not just faces)
- Easy to use with transformers library
- Good for general profile pictures
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO
import hashlib
import time
from tqdm import tqdm
import torch
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# GPU CONFIGURATION
# ============================================================

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

if CUDA_AVAILABLE:
    print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
else:
    print("⚠️ Running on CPU (no GPU detected)")

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
PROCESSED_DIR = DATA_DIR / "processed"

TIMEOUT = 10  # seconds for image download
MAX_RETRIES = 2
BATCH_SIZE = 64 if CUDA_AVAILABLE else 16  # Larger batches on GPU


# ============================================================
# 1. IMAGE DOWNLOADER
# ============================================================

def get_image_filename(url: str, profile_id: str, platform: str) -> str:
    """Generate unique filename for image based on URL hash."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{profile_id}_{platform}_{url_hash}.jpg"


def is_valid_url(url: str) -> bool:
    """Check if URL is valid and not empty."""
    if pd.isna(url) or not url or not isinstance(url, str):
        return False
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        return False
    return True


def download_image(url: str, save_path: str, timeout: int = TIMEOUT) -> Tuple[bool, str]:
    """
    Download image from URL and save to disk.
    
    Returns:
        Tuple[bool, str]: (success, message)
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}"
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type.lower():
            return False, f"Not an image: {content_type}"
        
        # Try to open as image to validate
        img_data = BytesIO(response.content)
        img = Image.open(img_data)
        
        # Convert to RGB if necessary (some images are RGBA, P mode, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save
        img.save(save_path, 'JPEG', quality=85)
        
        return True, "OK"
        
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection Error"
    except requests.exceptions.RequestException as e:
        return False, f"Request Error: {str(e)[:50]}"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"


def download_all_images(
    df: pd.DataFrame,
    output_dir: str = IMAGE_DIR,
    url_column: str = 'pictureURL',
    profile_id_column: str = 'profile_id',
    platform_column: str = 'platform'
) -> pd.DataFrame:
    """
    Download all images from DataFrame.
    
    Returns:
        DataFrame with added columns: image_path, download_status
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    print(f"Downloading images from {len(df)} profiles...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        url = row.get(url_column, '')
        profile_id = row.get(profile_id_column, f'unknown_{idx}')
        platform = row.get(platform_column, 'unknown')
        
        if not is_valid_url(url):
            results.append({
                'index': idx,
                'image_path': None,
                'download_status': 'Invalid URL'
            })
            continue
        
        # Generate filename
        filename = get_image_filename(url, profile_id, platform)
        save_path = os.path.join(output_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(save_path):
            results.append({
                'index': idx,
                'image_path': save_path,
                'download_status': 'Already exists'
            })
            continue
        
        # Download
        success, message = download_image(url, save_path)
        
        if success:
            results.append({
                'index': idx,
                'image_path': save_path,
                'download_status': 'OK'
            })
        else:
            results.append({
                'index': idx,
                'image_path': None,
                'download_status': message
            })
        
        # Be nice to servers
        time.sleep(0.1)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge with original
    df_result = df.copy()
    df_result['image_path'] = results_df['image_path'].values
    df_result['download_status'] = results_df['download_status'].values
    
    # Print summary
    success_count = (df_result['download_status'] == 'OK').sum()
    exists_count = (df_result['download_status'] == 'Already exists').sum()
    failed_count = len(df_result) - success_count - exists_count
    
    print(f"\n=== Download Summary ===")
    print(f"New downloads: {success_count}")
    print(f"Already existed: {exists_count}")
    print(f"Failed/Invalid: {failed_count}")
    
    return df_result


# ============================================================
# 2. EMBEDDING EXTRACTION (CLIP)
# ============================================================

class CLIPEmbedder:
    """
    Extract image embeddings using CLIP model with GPU acceleration.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model with GPU and FP16 optimization.
        
        Args:
            model_name: HuggingFace model name for CLIP
        """
        print(f"Loading CLIP model: {model_name}...")
        
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = DEVICE
        print(f"Using device: {self.device}")
        
        # ✨ แก้ไขตรงนี้: บังคับใช้ safetensors เพื่อหลีกเลี่ยงช่องโหว่ CVE-2025-32434
        try:
            self.model = CLIPModel.from_pretrained(
                model_name,
                use_safetensors=True  # 🔒 ใช้ไฟล์ปลอดภัย
            )
            print("✓ Loaded model with safetensors (secure)")
        except Exception as e:
            print(f"⚠️ Safetensors not available, trying default loading: {e}")
            self.model = CLIPModel.from_pretrained(model_name)
        
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Enable FP16 on GPU for 2x faster inference
        self.use_fp16 = CUDA_AVAILABLE
        if self.use_fp16:
            self.model = self.model.half()
            print("✓ CLIP loaded with FP16 (2x faster on GPU)")
        else:
            print("✓ CLIP loaded (CPU mode)")
    
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract embedding for a single image.
        
        Returns:
            numpy array of shape (512,) for CLIP-ViT-B/32
        """
        try:
            image = Image.open(image_path).convert('RGB')
            
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Use FP16 on GPU
            if self.use_fp16:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize embedding
            embedding = image_features.float().cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding from {image_path}: {e}")
            return None
    
    def extract_embeddings_batch(
        self,
        image_paths: List[str],
        batch_size: int = BATCH_SIZE
    ) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings for multiple images in batches.
        """
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting embeddings"):
            batch_paths = image_paths[i:i + batch_size]
            
            # Load images
            batch_images = []
            valid_indices = []
            
            for j, path in enumerate(batch_paths):
                if path and os.path.exists(path):
                    try:
                        img = Image.open(path).convert('RGB')
                        batch_images.append(img)
                        valid_indices.append(j)
                    except:
                        pass
            
            if not batch_images:
                embeddings.extend([None] * len(batch_paths))
                continue
            
            # Process batch
            try:
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Use FP16 on GPU for faster processing
                if self.use_fp16:
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                
                # Normalize
                batch_embeddings = image_features.float().cpu().numpy()
                batch_embeddings = batch_embeddings / np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                
                # Map back to original order
                result = [None] * len(batch_paths)
                for idx, emb in zip(valid_indices, batch_embeddings):
                    result[idx] = emb
                
                embeddings.extend(result)
                
            except Exception as e:
                print(f"Batch error: {e}")
                embeddings.extend([None] * len(batch_paths))
        
        return embeddings


def extract_and_save_embeddings(
    df: pd.DataFrame,
    output_dir: str = EMBEDDINGS_DIR,
    model_name: str = "openai/clip-vit-base-patch32"
) -> pd.DataFrame:
    """
    Extract embeddings for all images and save to disk.
    
    Returns:
        DataFrame with added embedding_path column
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter rows with valid images
    valid_mask = df['image_path'].notna() & (df['download_status'].isin(['OK', 'Already exists']))
    
    print(f"Valid images for embedding: {valid_mask.sum()} / {len(df)}")
    
    if valid_mask.sum() == 0:
        print("No valid images found for embedding extraction!")
        df['embedding_path'] = None
        return df
    
    # Initialize CLIP model
    embedder = CLIPEmbedder(model_name)
    
    # Extract embeddings
    image_paths = df.loc[valid_mask, 'image_path'].tolist()
    embeddings = embedder.extract_embeddings_batch(image_paths)
    
    # Save embeddings
    embedding_paths = [None] * len(df)
    
    valid_indices = df.index[valid_mask].tolist()
    
    for idx, emb in zip(valid_indices, embeddings):
        if emb is not None:
            profile_id = df.loc[idx, 'profile_id']
            platform = df.loc[idx, 'platform']
            
            emb_filename = f"{profile_id}_{platform}_embedding.npy"
            emb_path = os.path.join(output_dir, emb_filename)
            
            np.save(emb_path, emb)
            embedding_paths[df.index.get_loc(idx)] = emb_path
    
    df['embedding_path'] = embedding_paths
    
    # Summary
    success_count = sum(1 for p in embedding_paths if p is not None)
    print(f"\n=== Embedding Summary ===")
    print(f"Successfully extracted: {success_count}")
    print(f"Failed: {valid_mask.sum() - success_count}")
    
    return df


# ============================================================
# 3. PROCESS EXISTING IMAGES (NEW FUNCTION)
# ============================================================

def process_existing_images(
    input_csv: str = None,
    output_csv: str = None,
    images_dir: str = IMAGE_DIR,
    embeddings_dir: str = EMBEDDINGS_DIR
) -> pd.DataFrame:
    """
    Process existing images (skip download, only extract embeddings).
    
    This function:
    1. Loads the CSV
    2. Scans the images folder
    3. Matches images to profiles
    4. Extracts embeddings for all found images
    """
    # Set default paths
    if input_csv is None:
        input_csv = os.path.join(PROCESSED_DIR, "all_profiles_cleaned.csv")
    if output_csv is None:
        output_csv = os.path.join(PROCESSED_DIR, "profiles_with_embeddings.csv")
    
    print("=" * 60)
    print("PROCESS EXISTING IMAGES - EMBEDDING ONLY")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(input_csv)
    print(f"   Loaded {len(df)} profiles")
    
    # Scan images folder
    print(f"\n2. Scanning images folder: {images_dir}")
    if not os.path.exists(images_dir):
        print(f"   ERROR: Images folder not found: {images_dir}")
        return df
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   Found {len(image_files)} images")
    
    # Match images to profiles
    print("\n3. Matching images to profiles...")
    image_paths = []
    
    for idx, row in df.iterrows():
        profile_id = row.get('profile_id', f'unknown_{idx}')
        platform = row.get('platform', 'unknown')
        
        # Try to find matching image
        matched_image = None
        for img_file in image_files:
            # Check if filename contains profile_id and platform
            if f"{profile_id}_{platform}" in img_file:
                matched_image = os.path.join(images_dir, img_file)
                break
        
        image_paths.append(matched_image)
    
    df['image_path'] = image_paths
    df['download_status'] = df['image_path'].apply(lambda x: 'Already exists' if x else 'No image found')
    
    matched_count = sum(1 for p in image_paths if p is not None)
    print(f"   Matched {matched_count} / {len(df)} profiles to images")
    
    # Extract embeddings
    print("\n4. Extracting embeddings for all matched images...")
    df = extract_and_save_embeddings(df, output_dir=embeddings_dir)
    
    # Save results
    print(f"\n5. Saving results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    
    return df


# ============================================================
# 4. MAIN PIPELINE (ORIGINAL)
# ============================================================

def run_image_embedding_pipeline(
    input_csv: str = None,
    output_csv: str = None,
    images_dir: str = IMAGE_DIR,
    embeddings_dir: str = EMBEDDINGS_DIR,
    sample_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Run the complete image download and embedding pipeline.
    
    Args:
        input_csv: Path to input CSV with profile data
        output_csv: Path to save output CSV
        images_dir: Directory to save downloaded images
        embeddings_dir: Directory to save embeddings
        sample_size: If set, only process this many profiles (for testing)
    """
    # Set default paths using PROCESSED_DIR
    if input_csv is None:
        input_csv = os.path.join(PROCESSED_DIR, "all_profiles_cleaned.csv")
    if output_csv is None:
        output_csv = os.path.join(PROCESSED_DIR, "profiles_with_embeddings.csv")
    
    print("=" * 60)
    print("IMAGE DOWNLOAD & EMBEDDING PIPELINE")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(input_csv)
    print(f"   Loaded {len(df)} profiles")
    
    # Sample if needed
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"   Sampled {len(df)} profiles for testing")
    
    # Download images
    print("\n2. Downloading images...")
    df = download_all_images(df, output_dir=images_dir)
    
    # Extract embeddings
    print("\n3. Extracting embeddings...")
    df = extract_and_save_embeddings(df, output_dir=embeddings_dir)
    
    # Save results
    print(f"\n4. Saving results to {output_csv}...")
    df.to_csv(output_csv, index=False)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    
    return df


# ============================================================
# USAGE
# ============================================================

if __name__ == "__main__":
    print("Starting Image Embedding Pipeline...")
    print("NOTE: First run will download CLIP model (~350MB)")
    print()
    
    # ========================================
    # MODE SELECTION
    # ========================================
    SKIP_DOWNLOAD = True  # 🔧 เปลี่ยนเป็น True เพื่อข้ามการดาวน์โหลด
    
    if SKIP_DOWNLOAD:
        print("📁 MODE: Process existing images only (skip download)")
        df = process_existing_images()
    else:
        print("🌐 MODE: Download images + extract embeddings")
        df = run_image_embedding_pipeline(
            sample_size=None  # Full run
        )
    
    print("\nSample of results:")
    print(df[['profile_id', 'platform', 'download_status', 'embedding_path']].head(10))
