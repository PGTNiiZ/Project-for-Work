"""
Fix corrupted numpy — Run from terminal AFTER closing ALL Jupyter kernels.

วิธีใช้:
1. ปิด Jupyter Kernel ทั้งหมดก่อน (Kernel > Shut Down All Kernels)
2. เปิด Terminal (PowerShell) แล้วรัน:
   cd D:\66070260-Year3_Term2\Project1\Code\Project-for-Work
   python fix_numpy.py
3. เปิด notebook ใหม่แล้วรันได้เลย
"""
import shutil
import site
import sys
import subprocess
from pathlib import Path

print("=" * 60)
print("FIXING NUMPY (ปิด Jupyter Kernel ก่อนรัน!)")
print("=" * 60)

sp = Path(site.getusersitepackages())
print(f"Site-packages: {sp}")

# Step 1: Remove numpy
print("\n[1] Removing numpy...")
for item in sorted(sp.glob("numpy*")):
    try:
        if item.is_dir():
            shutil.rmtree(item)
            print(f"  ✅ Removed {item.name}/")
        else:
            item.unlink()
            print(f"  ✅ Removed {item.name}")
    except PermissionError as e:
        print(f"  ❌ LOCKED: {item.name} — Jupyter kernel is still running!")
        print(f"     ⚠️ กรุณาปิด Jupyter Kernel ก่อนแล้วรัน script นี้ใหม่")
        sys.exit(1)
    except Exception as e:
        print(f"  ⚠️ {item.name}: {e}")

# Remove numpy.libs too
nl = sp / "numpy.libs"
if nl.exists():
    shutil.rmtree(nl, ignore_errors=True)

# Verify removed
if (sp / "numpy").exists():
    print("\n  ❌ numpy ยังลบไม่ได้ — ปิด Jupyter แล้วลองใหม่")
    sys.exit(1)
else:
    print("\n  ✅ numpy removed successfully!")

# Step 2: Install numpy
print("\n[2] Installing numpy...")
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "numpy==2.0.2", "--user", "--no-cache-dir"],
    capture_output=True, text=True, timeout=300
)
if r.returncode == 0:
    print("  ✅ numpy 2.0.2 installed!")
else:
    print(f"  ⚠️ numpy 2.0.2 failed, trying latest...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy", "--user", "--no-cache-dir"],
        capture_output=True, text=True, timeout=300
    )
    if r.returncode == 0:
        print("  ✅ numpy (latest) installed!")
    else:
        print(f"  ❌ Failed: {r.stderr[-300:]}")
        sys.exit(1)

# Step 3: Verify numpy
print("\n[3] Verifying numpy...")
r = subprocess.run(
    [sys.executable, "-c", "import numpy; print(f'numpy {numpy.__version__} ✅')"],
    capture_output=True, text=True
)
if r.returncode == 0:
    print(f"  {r.stdout.strip()}")
else:
    print(f"  ❌ Verify failed: {r.stderr[-200:]}")
    sys.exit(1)

# Step 4: Install dependent packages
print("\n[4] Installing remaining packages...")
pkgs = [
    "pandas", "matplotlib", "seaborn", "scikit-learn",
    "faiss-cpu", "sentence-transformers", "keybert"
]
for pkg in pkgs:
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "--user", "-q"],
        capture_output=True, text=True, timeout=300
    )
    s = "✅" if r.returncode == 0 else "❌"
    print(f"  {s} {pkg}")

# Step 5: Final check
print("\n[5] Final verification...")
test = """
import numpy as np; print(f'  numpy {np.__version__}')
import pandas as pd; print(f'  pandas {pd.__version__}')
import faiss; print(f'  faiss OK')
from sentence_transformers import SentenceTransformer; print(f'  sentence-transformers OK')
from keybert import KeyBERT; print(f'  keybert OK')
import sklearn; print(f'  scikit-learn OK')
import matplotlib; print(f'  matplotlib OK')
"""
r = subprocess.run([sys.executable, "-c", test], capture_output=True, text=True)
if r.returncode == 0:
    print(r.stdout)
    print("=" * 60)
    print("🎉 ALL FIXED! เปิด notebook ได้เลย!")
    print("=" * 60)
else:
    print(f"  ⚠️ Some packages still have issues")
    print(r.stderr[-300:])
