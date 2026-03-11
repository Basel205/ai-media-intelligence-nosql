# AI-Powered Media Intelligence System
### Semantic Image Retrieval Using MongoDB Vector Search & CLIP Embeddings

A NoSQL course project demonstrating document-oriented storage, vector embeddings, hybrid queries, and a custom re-ranking algorithm for semantic image search.

---

## Project Overview

Modern users accumulate thousands of images with meaningless filenames (`IMG_2847.jpg`, `photo_final(2).jpg`). Traditional file systems rely on folder structure and manual tagging — both fail at scale.

This system solves that by letting you search images using natural language:

> *"sunset over water"* → returns the most visually relevant images, even with zero labels or tags.

The system uses **CLIP** (OpenAI's Contrastive Language-Image Pretraining model) to convert both images and text into 512-dimensional vectors, stores them in **MongoDB**, and retrieves semantically similar images via vector similarity search.

---

## Novelty — Composite Semantic Similarity (CSS)

Standard vector search ranks images purely by cosine similarity. This has a flaw: a blurry, dark, low-quality image can outrank a sharp one if its embedding is marginally closer to the query.

This project introduces **Composite Semantic Similarity (CSS)**, a custom re-ranking algorithm that combines three signals:

```
CSS(query, image) = α · cosine_similarity(query, image)
                 + β · sharpness_score(image)
                 + γ · brightness_score(image)

Default weights: α = 0.60, β = 0.25, γ = 0.15
```

Where:
- **cosine_similarity** — CLIP semantic match between text query and image embedding
- **sharpness_score** — normalised Laplacian variance (blur detection)
- **brightness_score** — tent function penalising overexposed and underexposed images

CSS is benchmarked against the pure cosine baseline using **Precision@5** and **Precision@10** across 6 query categories, demonstrating measurable improvement in retrieval quality.

---

## Architecture

```
Images on Disk (any folder)
        │
        ▼
┌─────────────────┐
│  Image Scanner  │  Recursively finds all image files
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Metadata Extractor  │  Width, height, blur score, brightness
└────────┬────────────┘
         │
         ▼
┌──────────────────┐
│    MongoDB       │  Stores document per image
│  media_intel-    │  { file_path, metadata, embeddings }
│  ligence DB      │
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│   CLIP Embedder      │  Generates 512-dim vector per image
│   (ViT-B/32)         │  Stored back into MongoDB
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────────────┐
│         Search Engine            │
│                                  │
│  Baseline: pure cosine similarity│
│  CSS: cosine + quality re-rank   │
│  Hybrid: MongoDB filter + vector │
└────────┬─────────────────────────┘
         │
         ▼
┌──────────────────┐
│   Flask Dashboard │  Web UI at localhost:5000
└──────────────────┘
```

---

## Why NoSQL?

| Requirement | SQL | MongoDB (this project) |
|---|---|---|
| Store 512-float vector per row | ❌ No native support | ✅ Native array storage |
| Flexible schema (add fields later) | ❌ Requires migration | ✅ Schema-free documents |
| Hybrid filter + vector ranking | ❌ Not possible | ✅ Single pipeline query |
| Heterogeneous image metadata | ❌ Rigid columns | ✅ Nested documents |
| Scale horizontally | ❌ Complex sharding | ✅ Built-in |

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Database | MongoDB (local) |
| AI Model | CLIP ViT-B/32 (OpenAI via HuggingFace) |
| DB Driver | PyMongo |
| Image Processing | OpenCV, Pillow |
| Web Framework | Flask |
| ML Framework | PyTorch, Transformers |

---

## Project Structure

```
media-intelligence-nosql/
│
├── database/
│   ├── mongo_connection.py      # MongoDB connection with error handling
│   ├── schema_setup.py          # Collection + index creation
│   └── media_repository.py     # All CRUD operations (single source of truth)
│
├── ingestion/
│   ├── scanner.py               # Recursive image file discovery
│   ├── metadata_extractor.py    # Blur score, brightness, resolution extraction
│   └── pipeline.py              # Full ingestion pipeline with duplicate check
│
├── embeddings/
│   ├── clip_model.py            # CLIP ViT-B/32 wrapper (image + text)
│   └── generate_embeddings.py  # Batch embedding generation
│
├── search/
│   ├── semantic_search.py       # Baseline: pure cosine similarity
│   ├── hybrid_search.py         # CSS: composite re-ranking (novelty)
│   ├── metadata_filters.py      # MongoDB metadata pre-filtering
│   └── evaluation.py            # Precision@K benchmark (baseline vs CSS)
│
├── quality/
│   ├── duplicate_detector.py    # Near-duplicate detection via cosine threshold
│   └── image_quality.py         # Quality tier tagging
│
├── utils/
│   ├── config.py                # Central configuration (paths, weights, thresholds)
│   └── file_utils.py            # File extension helpers
│
├── scripts/
│   ├── ingest_dataset.py        # Standalone ingestion script
│   └── run_search_demo.py       # CLI demo: baseline vs CSS comparison
│
├── templates/
│   └── index.html               # Dashboard UI (search, compare, evaluate)
│
├── app.py                       # Flask web application
├── main.py                      # CLI entry point
└── requirements.txt
```

---

## Setup & Installation

### Prerequisites
- Python 3.11+
- MongoDB running locally on port 27017
- ~2GB disk space (CLIP model cache)

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure your images folder
Open `utils/config.py` and set:
```python
DATASET_PATH = r"C:\path\to\your\images"
```

---

## Usage

### CLI

```bash
# 1. Initialise database schema
python main.py setup

# 2. Ingest images from configured folder
python main.py ingest

# 3. Generate CLIP embeddings (run once, takes a few minutes)
python main.py embed

# 4. Check status
python main.py stats

# 5. Search via CLI
python main.py search "mountain landscape"
python main.py search "mountain landscape" --baseline   # compare methods
python main.py compare "sea waves"                      # side-by-side

# 6. Find near-duplicate images
python main.py duplicates

# 7. Run evaluation benchmark
python main.py evaluate
```

### Web Dashboard
```bash
python app.py
# Open http://localhost:5000
```

The dashboard has three tabs:
- **Results** — search with CSS or baseline, with quality filter sliders
- **Baseline vs CSS** — side-by-side comparison for the same query
- **Evaluation** — live Precision@K benchmark table

---

## MongoDB Document Schema

Each image is stored as a single document:

```json
{
  "file_id": "uuid-string",
  "file_path": "C:/Users/.../image.jpg",
  "file_name": "image.jpg",
  "media_type": "image",
  "metadata": {
    "width": 1920,
    "height": 1080,
    "brightness": 0.61,
    "blur_score": 423.7,
    "aspect_ratio": 1.778
  },
  "embeddings": {
    "clip_image": [0.023, -0.141, 0.334, "...512 floats total"]
  },
  "quality_flags": {
    "is_blurry": false,
    "is_dark": false,
    "quality_tier": "high"
  }
}
```

Schema evolution is demonstrated by the addition of `quality_flags` — added retroactively to existing documents with no migration required, which would be impossible in a relational database without an `ALTER TABLE` operation.

---

## Evaluation Results

Precision@K comparison — Baseline (cosine) vs CSS across 6 query categories:

| Query | P@5 Baseline | P@5 CSS | Δ@5 | P@10 Baseline | P@10 CSS | Δ@10 |
|---|---|---|---|---|---|---|
| urban buildings and architecture | 0.800 | 0.800 | +0.000 | 0.800 | 0.700 | -0.100 |
| dense forest with trees | 1.000 | 1.000 | +0.000 | 0.900 | 1.000 | +0.100 |
| glacier ice and snow | 1.000 | 0.800 | -0.200 | 1.000 | 0.900 | -0.100 |
| mountain peak landscape | 0.400 | 0.800 | **+0.400** | 0.400 | 0.700 | **+0.300** |
| sea ocean water waves | 1.000 | 1.000 | +0.000 | 0.900 | 0.900 | +0.000 |
| street road city traffic | 0.600 | 1.000 | **+0.400** | 0.700 | 1.000 | **+0.300** |
| **Average** | **0.800** | **0.900** | **+0.100** | **0.783** | **0.867** | **+0.083** |

CSS improves average Precision@5 by **+0.100** and Precision@10 by **+0.083** over the pure cosine baseline.
The largest gains are on ambiguous categories (mountain, street) where image quality acts as a meaningful tiebreaker.

---

## Limitations

- **Approximate search**: cosine similarity is computed in-memory (Python), not via a dedicated ANN index. Acceptable for this dataset scale; production deployments would use MongoDB Atlas Vector Search or FAISS.
- **CLIP limitations**: CLIP may struggle with highly abstract or domain-specific images (medical scans, diagrams, etc.).
- **CSS weights are static**: weights (α, β, γ) are manually tuned. A learned weighting via user feedback would be a natural extension.
- **Local only**: images must be present on the same machine as the server.

---

## Possible Extensions

- MongoDB Atlas Vector Search for ANN indexing at scale
- User feedback loop to tune CSS weights per user
- Face detection count as an additional metadata signal
- Location estimation from image content
- REST API for mobile client integration