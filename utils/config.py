import os

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI     = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DB_NAME",   "media_intelligence")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.getenv("DATASET_PATH", os.path.join(BASE_DIR, "intel_dataset", "seg_test", "seg_test"))

# ── Watched folders (system-level watcher) ────────────────────────────────────
WATCHED_FOLDERS = [
    r"C:\Users\kbase\Downloads",
    r"C:\Users\kbase\OneDrive\Desktop",
    r"C:\Users\kbase\OneDrive\Documents",
    r"C:\Users\kbase\OneDrive\Pictures",
    r"C:\Users\kbase\Videos",
]

# Folders to skip during scan (system/app junk)
EXCLUDED_DIRS = {
    "windows", "program files", "program files (x86)",
    "appdata", "$recycle.bin", "programdata",
    "node_modules", ".git", "__pycache__",
    "system volume information", "recovery",
}

# Minimum file size to embed (skip icons/thumbnails)
MIN_IMAGE_SIZE_BYTES = 50 * 1024   # 50 KB

# ── CLIP ──────────────────────────────────────────────────────────────────────
CLIP_MODEL    = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512

# ── Search ────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 20

# Composite Semantic Similarity weights — must sum to 1.0
CSS_ALPHA    = 0.60
CSS_BETA     = 0.25
CSS_GAMMA    = 0.15
CSS_MAX_BLUR = 2000.0

# ── Quality ───────────────────────────────────────────────────────────────────
DUPLICATE_THRESHOLD = 0.97
MIN_BLUR_SCORE      = 50.0
MIN_BRIGHTNESS      = 0.15

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_PORT = 5000
FLASK_HOST = "127.0.0.1"
