import os

# ── MongoDB ───────────────────────────────────────────────────────────────────
MONGO_URI     = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DB_NAME",   "media_intelligence")

# ── Dataset ───────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATASET_PATH = os.getenv("DATASET_PATH", r"C:\Users\kbase\Projects\media-intelligence-nosql\data\raw_media")
DATASET_PATH = os.getenv("DATASET_PATH", os.path.join(BASE_DIR, "intel_dataset", "seg_test", "seg_test"))


# ── CLIP ──────────────────────────────────────────────────────────────────────
CLIP_MODEL    = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512

# ── Search ────────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 10

# Composite Semantic Similarity weights — must sum to 1.0
CSS_ALPHA = 0.60   # CLIP cosine similarity  (semantic relevance)
CSS_BETA  = 0.25   # Image sharpness         (blur score)
CSS_GAMMA = 0.15   # Image brightness        (exposure quality)
CSS_MAX_BLUR = 2000.0   # Laplacian variance normalisation ceiling

# ── Quality ───────────────────────────────────────────────────────────────────
DUPLICATE_THRESHOLD = 0.97
MIN_BLUR_SCORE      = 50.0
MIN_BRIGHTNESS      = 0.15
