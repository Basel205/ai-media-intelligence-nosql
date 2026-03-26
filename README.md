# AI-Powered Media Intelligence System

> A local, privacy-first semantic image search engine that understands your photos — not just their filenames.

---

## What Is This?

You have thousands of photos scattered across your laptop. Their filenames are `IMG_2847.jpg`, `screenshot_final(2).png`, `photo_20231104_183422.jpg`. Searching for "beach sunset" or "birthday party" is impossible — there are no tags, no labels, no metadata that means anything.

This system fixes that. It reads every image on your machine, uses an AI model to understand what's actually *in* each photo, and lets you search your entire photo library using plain English — exactly like Google Photos, but running 100% locally. Your images never leave your machine.

```
You type:  "mountain with snow"
System returns: the 20 most visually relevant photos from anywhere on your laptop
```

No cloud. No API keys. No privacy trade-offs.

---

## Key Features

- **Natural language search** — search by meaning, not filename
- **System tray integration** — runs silently in the background, always ready
- **Automatic indexing** — new images added anywhere in your watched folders get indexed automatically
- **GPU-accelerated** — uses your NVIDIA GPU for fast embedding generation
- **Dual search interface** — desktop app window (native) + web dashboard (browser)
- **CSS novelty algorithm** — custom re-ranking that factors in image quality alongside semantic similarity
- **Hybrid MongoDB queries** — combine semantic vector search with structured metadata filters
- **Duplicate detection** — finds near-identical images using embedding similarity
- **Quality-aware results** — blurry and dark images are ranked lower automatically

---

## How It Works

### The Core Idea — CLIP Embeddings

This system uses **CLIP** (Contrastive Language-Image Pretraining), an AI model trained by OpenAI on 400 million image-text pairs. CLIP's key property: it maps both images and text into the same 512-dimensional vector space.

This means:
- A photo of a forest → `[0.02, -0.14, 0.33, ...]` (512 numbers)
- The text "forest trees" → `[0.01, -0.12, 0.31, ...]` (512 numbers)
- These two vectors are **close to each other** in the vector space

Finding similar images becomes a mathematical problem: find all image vectors closest to your query vector. This is cosine similarity search.

### The Full Pipeline

```
Your photos (stay on disk — never copied or moved)
        │
        ▼
Scanner finds all .jpg/.png/.webp files across watched folders
        │
        ▼
Metadata extracted per image:
  - Resolution (width × height)
  - Sharpness (Laplacian variance — blur detection)
  - Brightness (HSV colour space mean)
  - Aspect ratio
        │
        ▼
CLIP model generates 512-dimensional embedding per image
(runs on your GPU via CUDA — fast)
        │
        ▼
MongoDB stores the document:
  { file_path, metadata, embeddings: [512 floats] }
        │
        ▼
You search "mountain sunset"
        │
        ▼
CLIP converts your text → 512-number vector
Compare against all stored image vectors
Apply CSS re-ranking (see below)
        │
        ▼
Top matches returned — images served from their original disk location
```

### Why MongoDB Is the Right Database Here

This project uses MongoDB deliberately — not just because it's a NoSQL course project, but because the problem genuinely requires what MongoDB provides:

| Requirement | SQL | MongoDB |
|---|---|---|
| Store 512-float vector per image | ❌ Needs separate table, 512 rows per image | ✅ Native array in document |
| Flexible schema — add fields later | ❌ `ALTER TABLE` required, locks table | ✅ Just add the field, no migration |
| Hybrid filter + vector ranking | ❌ Fundamentally not possible | ✅ Single query pipeline |
| Nested metadata (variable per image) | ❌ Rigid columns, nulls everywhere | ✅ Nested documents, any shape |
| Horizontal scaling | ❌ Complex sharding setup | ✅ Built-in |

**Schema evolution in practice:** Images are first ingested with just `metadata`. Embeddings are added in a second pass. Quality flags are added later. Every document has a different set of fields at different points in time — MongoDB handles this naturally. SQL would require three `ALTER TABLE` operations and careful migration scripts.

---

## The Novelty — Composite Semantic Similarity (CSS)

Standard vector search has a real flaw: it ranks images purely by semantic closeness. A blurry, underexposed, low-quality photo of a mountain can outrank a sharp, well-lit one if its CLIP embedding happens to be marginally closer to the query vector. Semantic relevance and visual quality are independent — standard search ignores quality entirely.

**CSS** is a custom re-ranking algorithm that addresses this:

```
CSS(query, image) = α · cosine_similarity(query_vector, image_vector)
                 + β · sharpness_score(image)
                 + γ · brightness_score(image)

Default weights:  α = 0.60   β = 0.25   γ = 0.15
```

**Sharpness score** is derived from the Laplacian variance of the image — a standard computer vision metric for blur detection. Higher variance = sharper image. Normalised to [0, 1] with an empirical ceiling of 2000.

**Brightness score** uses a tent function peaking at 0.5 (ideal exposure):
```python
brightness_score = max(0, 1 - abs(brightness - 0.5) * 2)
```
This penalises both underexposed (dark) and overexposed (washed out) images symmetrically.

All three signals are stored in MongoDB at ingest time — no additional computation at search time.

### Evaluation Results

CSS is benchmarked against the pure cosine baseline using **Precision@K** — "of the top K results for a given query, what fraction are actually from the correct category?" Intel dataset folder labels (buildings, forest, glacier, mountain, sea, street) serve as ground truth exclusively for this measurement.

| Query | P@5 Baseline | P@5 CSS | Δ@5 | P@10 Baseline | P@10 CSS | Δ@10 |
|---|---|---|---|---|---|---|
| urban buildings and architecture | 0.800 | 0.800 | +0.000 | 0.800 | 0.700 | -0.100 |
| dense forest with trees | 1.000 | 1.000 | +0.000 | 0.900 | 1.000 | +0.100 |
| glacier ice and snow | 1.000 | 0.800 | -0.200 | 1.000 | 0.900 | -0.100 |
| mountain peak landscape | 0.400 | 0.800 | **+0.400** | 0.400 | 0.700 | **+0.300** |
| sea ocean water waves | 1.000 | 1.000 | +0.000 | 0.900 | 0.900 | +0.000 |
| street road city traffic | 0.600 | 1.000 | **+0.400** | 0.700 | 1.000 | **+0.300** |
| **Average** | **0.800** | **0.900** | **+0.100** | **0.783** | **0.867** | **+0.083** |

CSS improves average Precision@5 by **+10 percentage points** and Precision@10 by **+8.3 percentage points**. The largest gains are on ambiguous categories (mountain, street) where image quality acts as a meaningful tiebreaker between semantically similar candidates.

---

## System Architecture

### Threading Model

```
start_debug.py / start.pyw
        │
        ▼
TrayApp.start()
        │
        ├── Main Thread ──────── tkinter SearchWindow (required by tkinter)
        │
        ├── Thread 2 ─────────── pystray system tray icon + menu
        │
        ├── Thread 3 ─────────── Flask API (localhost:5000)
        │
        ├── Thread 4 ─────────── FolderWatcher (watchdog, monitors 5 folders)
        │
        └── Thread 5 ─────────── FullSystemScanner (background, auto on launch)
```

**Single shared CLIPEmbedder:** The CLIP model is loaded exactly once at startup (~3 seconds, ~600MB VRAM) and injected into every component that needs it — the scanner, watcher, Flask app, and search engines all share one instance via dependency injection.

### Monitored Folders

The system watches these folders by default (all subfolders included):

```
C:\Users\kbase\Downloads
C:\Users\kbase\OneDrive\Desktop
C:\Users\kbase\OneDrive\Documents
C:\Users\kbase\OneDrive\Pictures
C:\Users\kbase\Videos
```

System directories (`Windows\`, `AppData\`, `Program Files\`, etc.) and files under 50KB (icons, thumbnails) are automatically excluded.

---

## Project Structure

```
media-intelligence-nosql/
│
├── tray/
│   └── tray_app.py              # System tray orchestrator — boots all components
│
├── watcher/
│   └── folder_watcher.py        # Watchdog file system monitor — auto-indexes new images
│
├── ui/
│   └── search_window.py         # tkinter desktop search window
│
├── ingestion/
│   ├── scanner.py               # Recursive image file discovery
│   ├── metadata_extractor.py    # Blur score, brightness, resolution extraction
│   ├── pipeline.py              # Batch ingestion pipeline
│   └── full_scan.py             # Full system scan across all watched folders
│
├── embeddings/
│   ├── clip_model.py            # CLIP ViT-B/32 wrapper — image + text embedding
│   └── generate_embeddings.py   # Standalone batch embedding script
│
├── search/
│   ├── semantic_search.py       # Baseline: pure cosine similarity search
│   ├── hybrid_search.py         # CSS: composite quality-aware re-ranking
│   ├── metadata_filters.py      # MongoDB pre-filter builders
│   └── evaluation.py            # Precision@K benchmark (baseline vs CSS)
│
├── database/
│   ├── mongo_connection.py      # MongoDB connection with error handling
│   ├── schema_setup.py          # Collection and index creation
│   └── media_repository.py      # All CRUD operations — single source of truth
│
├── quality/
│   ├── duplicate_detector.py    # Near-duplicate detection via cosine threshold
│   └── image_quality.py         # Quality tier tagging (high / medium / low)
│
├── templates/
│   └── index.html               # Web dashboard (search + compare + evaluate tabs)
│
├── utils/
│   ├── config.py                # All configuration — paths, weights, thresholds
│   └── file_utils.py            # Supported image extension helpers
│
├── scripts/
│   ├── ingest_dataset.py        # Standalone ingestion script
│   └── run_search_demo.py       # CLI demo: baseline vs CSS side-by-side
│
├── intel_dataset/               # Intel Image Classification dataset (evaluation only)
│   └── seg_test/seg_test/       # 3000 labeled images across 6 categories
│
├── app.py                       # Flask REST API
├── main.py                      # CLI entry point
├── start.pyw                    # Silent production launcher (no terminal window)
├── start_debug.py               # Development launcher (terminal visible)
└── requirements.txt
```

---

## MongoDB Document Schema

Every image is stored as a single self-contained document:

```json
{
  "file_id":    "3f7a2b1c-...",
  "file_path":  "C:/Users/kbase/Pictures/holiday/beach.jpg",
  "file_name":  "beach.jpg",
  "media_type": "image",
  "metadata": {
    "width":        4032,
    "height":       3024,
    "brightness":   0.61,
    "blur_score":   812.4,
    "aspect_ratio": 1.333
  },
  "embeddings": {
    "clip_image": [0.023, -0.141, 0.334, "... 512 floats total"]
  },
  "quality_flags": {
    "is_blurry":      false,
    "is_dark":        false,
    "is_overexposed": false,
    "quality_tier":   "high"
  }
}
```

**The image file itself is never copied or moved.** MongoDB stores only the path, metadata, and the 2KB embedding vector. Storage overhead: ~2KB per image. For 50,000 photos, the entire MongoDB database is under 200MB.

**Indexes created:**
- `file_id` (unique) — fast document lookup
- `file_path` (unique) — duplicate check on ingest
- `metadata.blur_score` — used by hybrid search filters
- `metadata.brightness` — used by hybrid search filters
- `metadata.width` — used by resolution filters

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Database | MongoDB (local, port 27017) |
| AI Model | CLIP ViT-B/32 via HuggingFace Transformers |
| ML Framework | PyTorch 2.8.0 + CUDA 12.8 |
| GPU | NVIDIA RTX 4060 (auto-detected via `torch.cuda.is_available()`) |
| DB Driver | PyMongo 4.6.1 |
| Image Processing | OpenCV, Pillow |
| Web Framework | Flask |
| Desktop UI | tkinter (built-in Python) |
| System Tray | pystray |
| File Watching | watchdog |
| Environment | Anaconda (conda env: SIH) |

---

## Setup & Installation

### Prerequisites

- Python 3.11+ (Anaconda recommended)
- MongoDB running locally on port 27017
- NVIDIA GPU with CUDA drivers (CPU fallback works but is slower)

### Install Dependencies

```bash
conda activate SIH
pip install transformers pymongo flask pystray watchdog Pillow opencv-python torch torchvision tqdm
```

For CUDA-enabled PyTorch (if not already installed):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Configure Watched Folders

Open `utils/config.py` and update `WATCHED_FOLDERS` to match your machine:

```python
WATCHED_FOLDERS = [
    r"C:\Users\YourName\Downloads",
    r"C:\Users\YourName\Pictures",
    r"C:\Users\YourName\Desktop",
    r"C:\Users\YourName\Documents",
    r"C:\Users\YourName\Videos",
]
```

---

## Running the Application

### Full System (Recommended)

```bash
conda activate SIH
cd path/to/media-intelligence-nosql
python start_debug.py
```

On launch:
1. CLIP model loads on GPU (~3 seconds)
2. Flask API starts at `http://127.0.0.1:5000`
3. Folder watcher begins monitoring all configured folders
4. Background scan starts automatically across all watched folders
5. System tray icon appears — click to open search window

### Auto-Start on Windows Login

1. Press `Win+R`, type `shell:startup`, press Enter
2. Create a new shortcut with:
   - **Target:** `C:\path\to\anaconda3\envs\SIH\pythonw.exe start.pyw`
   - **Start in:** `C:\path\to\media-intelligence-nosql`

### CLI (Without Tray App)

```bash
python main.py setup              # Initialise database schema
python main.py ingest             # Ingest images from configured folder
python main.py embed              # Generate CLIP embeddings
python main.py stats              # Show database statistics
python main.py search "query"     # CSS search
python main.py search "query" --baseline   # Cosine baseline
python main.py compare "query"    # Side-by-side comparison
python main.py evaluate           # Run Precision@K benchmark
python main.py duplicates         # Find near-duplicate images
```

### Web Dashboard

```bash
python app.py
# Open http://localhost:5000
```

Three tabs:
- **Results** — search with quality filter sliders (min sharpness, min brightness)
- **Baseline vs CSS** — same query, both methods side by side
- **Evaluation** — live Precision@K benchmark table

---

## Limitations

- **In-memory vector search:** Cosine similarity is computed in Python across all documents. For the current dataset size this is fast; at 100k+ images, MongoDB Atlas Vector Search or FAISS would be needed for sub-second response.
- **Static CSS weights:** The α/β/γ weights are manually tuned. A learned weighting via user feedback would improve personalisation.
- **Local machine only:** Absolute file paths are stored. Moving images or running on a different machine requires re-indexing.
- **CLIP limitations:** Performance degrades on highly domain-specific images (medical scans, technical diagrams, handwritten text).

---

## Possible Extensions

- **Face recognition** — detect and cluster faces across the library, name people, enable "show photos of Sarah" search. Face embeddings stored as sub-document arrays inside image documents — a strong additional NoSQL justification.
- **MongoDB Atlas Vector Search** — replace in-memory cosine with native `$vectorSearch` operator for ANN indexing at scale
- **Adaptive CSS weights** — tune α/β/γ per user via implicit feedback (click-through data stored in MongoDB)
- **Mobile companion app** — REST API already exists, attach a mobile client
- **Video support** — extract keyframes, embed and index them alongside images
