# AI-Powered Media Intelligence System

> A local, privacy-first media intelligence system for semantic image search, quality-aware ranking, and face clustering, built on MongoDB.

---

## What This Project Does

This project indexes images from your local machine, understands their visual content using AI, stores rich metadata in MongoDB, and lets you search the collection in natural language.

It now supports two major intelligence layers:

- Semantic image understanding with CLIP embeddings
- Person discovery with face detection, face embeddings, clustering, and manual person labeling

Everything runs locally:

- Images remain on disk
- MongoDB stores paths, metadata, embeddings, and face sub-documents
- No cloud image upload is required

---

## Core Features

- Natural language image search using CLIP
- GPU-accelerated embedding generation
- Automatic indexing of newly added images
- Quality-aware reranking with Composite Semantic Similarity (CSS)
- MongoDB-backed hybrid search using metadata filters plus vector ranking
- Duplicate detection support
- Local Flask dashboard
- Local desktop search window
- Face detection with InsightFace
- Face clustering with DBSCAN
- People tab for browsing discovered clusters
- Manual naming of person clusters

---

## New Face Recognition / People Module

The project was extended with a face intelligence pipeline so the system can discover people across the image collection.

### New capability

For each indexed image, the system can now:

1. Detect faces
2. Extract a face embedding for each face
3. Store those faces inside the image document in MongoDB
4. Cluster similar faces into person groups
5. Let the user browse those clusters in the UI
6. Let the user assign a human-readable name to a cluster

### New modules

- `faces/face_detector.py`
- `faces/face_clusterer.py`
- `faces/__init__.py`

### Face detection model

Face detection and face embedding extraction are performed using InsightFace with the `buffalo_l` model.

This provides:

- face bounding boxes
- face confidence score
- normalized ArcFace-style face embeddings

### Face clustering

Detected faces are clustered using DBSCAN over cosine distance between face embeddings.

Why DBSCAN:

- number of people does not need to be known in advance
- singleton or bad detections can be treated as noise
- works well for face-embedding clustering

Cluster output:

- real clusters are stored as IDs like `cluster_0000`, `cluster_0001`
- unmatched faces are marked as `"noise"`

---

## NoSQL Design and MongoDB Schema

This project is a strong MongoDB / NoSQL use case because each image document evolves over time and stores multiple nested intelligence layers.

### Existing schema before face work

Each image already stored:

- basic identity fields
- file path
- image metadata
- CLIP embedding
- quality flags

Example:

```json
{
  "file_id": "3f7a2b1c-...",
  "file_path": "C:/Users/kbase/Pictures/holiday/beach.jpg",
  "file_name": "beach.jpg",
  "media_type": "image",
  "metadata": {
    "width": 4032,
    "height": 3024,
    "brightness": 0.61,
    "blur_score": 812.4,
    "aspect_ratio": 1.333
  },
  "embeddings": {
    "clip_image": [0.023, -0.141, 0.334]
  },
  "quality_flags": {}
}
```

### Face schema added in this update

A new top-level `faces` array was added to each media document.

Each element is a nested face sub-document:

```json
{
  "face_id": "uuid",
  "bbox": [x1, y1, x2, y2],
  "embedding": [0.12, -0.04, 0.88],
  "det_score": 0.98,
  "cluster_id": "cluster_0003",
  "person_label": "Sarah"
}
```

Full image document after this update:

```json
{
  "file_id": "3f7a2b1c-...",
  "file_path": "C:/Users/kbase/Pictures/holiday/beach.jpg",
  "file_name": "beach.jpg",
  "media_type": "image",
  "metadata": {
    "width": 4032,
    "height": 3024,
    "brightness": 0.61,
    "blur_score": 812.4,
    "aspect_ratio": 1.333
  },
  "embeddings": {
    "clip_image": [0.023, -0.141, 0.334]
  },
  "quality_flags": {},
  "faces": [
    {
      "face_id": "4f3d3a0d-31b9-40fa-8b17-2a8a517e4c01",
      "bbox": [120, 45, 262, 211],
      "embedding": [0.12, -0.04, 0.88],
      "det_score": 0.98,
      "cluster_id": "cluster_0003",
      "person_label": "Sarah"
    }
  ]
}
```

### Meaning of face states in MongoDB

- `faces` field missing:
  the document has not yet been face-processed
- `"faces": []`:
  the document has been processed and no faces were found
- `cluster_id = null`:
  the face was detected but has not yet been clustered
- `cluster_id = "noise"`:
  the face was processed but did not belong to a stable cluster
- `person_label = null`:
  no manual label has been assigned yet

### Why MongoDB fits especially well

MongoDB works well here because:

- image documents can evolve over time without schema migrations
- CLIP embeddings are stored as arrays directly in the document
- face arrays model one-to-many image-to-face relationships naturally
- nested sub-documents avoid SQL joins
- array filters let us update specific faces inside a document

---

## Processing Pipeline

### Semantic image pipeline

```text
Image on disk
  -> metadata extraction
  -> CLIP image embedding
  -> MongoDB storage
  -> natural language text query
  -> CLIP text embedding
  -> cosine similarity search
  -> CSS reranking
```

### Face pipeline

```text
Indexed image
  -> InsightFace detects faces
  -> one face embedding per detected face
  -> save faces[] inside MongoDB image document
  -> full-scan or manual recluster
  -> DBSCAN groups similar faces
  -> write cluster_id back to each face sub-document
  -> UI shows one representative face per cluster
  -> user can assign person_label to the cluster
```

### Backfill behavior for old indexed images

One major part of this update was support for backfilling faces onto already indexed MongoDB documents.

Problem solved:

- initially, only newly inserted images got face detection
- older indexed images were skipped
- therefore the People tab could show zero clusters even though the database already had many images

Current behavior:

- new images get faces detected during ingest
- old documents that do not yet have a `faces` field are face-processed during full scan
- images with no faces are explicitly saved as `faces: []`

This makes the NoSQL state complete and prevents endless reprocessing of non-face images.

---

## Search and Ranking

### CLIP embeddings

The system uses CLIP to place images and text into the same vector space. Search becomes a cosine-similarity problem between:

- query text embedding
- stored image embeddings

### CSS reranking

Composite Semantic Similarity (CSS) improves ranking by combining:

- semantic cosine similarity
- sharpness score
- brightness score

Formula:

```text
CSS(query, image) =
    alpha * cosine_similarity
  + beta  * sharpness_score
  + gamma * brightness_score
```

This helps rank visually better images above semantically similar but poor-quality ones.

---

## Application Architecture

### Runtime components

- `tray/tray_app.py`
  system tray orchestrator
- `watcher/folder_watcher.py`
  automatic indexing of new files
- `ingestion/full_scan.py`
  full library scan and face backfill
- `app.py`
  Flask API and browser UI
- `ui/search_window.py`
  tkinter desktop search UI

### Important folders

```text
media-intelligence-nosql/
|
|-- app.py
|-- main.py
|-- start.pyw
|-- start_debug.py
|
|-- database/
|   |-- mongo_connection.py
|   |-- schema_setup.py
|   `-- media_repository.py
|
|-- embeddings/
|   |-- clip_model.py
|   `-- generate_embeddings.py
|
|-- faces/
|   |-- __init__.py
|   |-- face_detector.py
|   `-- face_clusterer.py
|
|-- ingestion/
|   |-- scanner.py
|   |-- metadata_extractor.py
|   |-- pipeline.py
|   `-- full_scan.py
|
|-- search/
|   |-- semantic_search.py
|   |-- hybrid_search.py
|   |-- metadata_filters.py
|   `-- evaluation.py
|
|-- templates/
|   `-- index.html
|
|-- tray/
|   `-- tray_app.py
|
|-- ui/
|   `-- search_window.py
|
`-- watcher/
    `-- folder_watcher.py
```

---

## API Endpoints

### Existing endpoints

- `GET /api/search`
- `GET /api/compare`
- `GET /api/evaluate`
- `GET /api/stats`
- `GET /image`

### New people / face endpoints

- `GET /api/people`
  returns cluster representatives for the People grid
- `GET /api/people/<cluster_id>`
  returns all images/faces in that cluster
- `POST /api/people/<cluster_id>/label`
  applies a human-readable name to the cluster
- `POST /api/people/recluster`
  reruns face clustering
- `GET /face`
  dynamically crops and serves a face from the original image using the stored bounding box

---

## UI Additions

The web dashboard now has a fourth tab:

- Results
- Baseline vs CSS
- Evaluation
- People

### People tab behavior

- shows one representative face thumbnail per cluster
- shows cluster size
- opens cluster detail view on click
- shows all matching images/faces in that cluster
- supports manual labeling of a cluster
- supports manual reclustering

The system tray also now has a face-related action:

- `Recluster Faces`

---

## Repository / Data Access Changes

The repository layer in `database/media_repository.py` was extended to support face-aware NoSQL operations.

### New repository methods

- `upsert_faces(file_path, faces)`
- `get_all_faces()`
- `update_face_clusters(face_to_cluster)`
- `update_person_label(cluster_id, person_label)`
- `search_by_person(person_label)`
- `get_faces_by_cluster(cluster_id)`
- `remove_face_from_cluster(face_id)`
- `count_face_clusters()`
- `needs_face_processing(file_path)`

### MongoDB-specific techniques used

- nested array storage for `faces`
- `array_filters` to update selected face elements
- distinct cluster counting using nested `faces.cluster_id`
- state detection via existence / absence of `faces`

---

## Setup

### Prerequisites

- Python 3.11+
- MongoDB running locally on port `27017`
- Conda environment recommended
- NVIDIA GPU optional but recommended

### Install dependencies

```bash
conda activate SIH
pip install transformers pymongo flask pystray watchdog Pillow opencv-python torch torchvision tqdm
pip install insightface scikit-learn onnxruntime-gpu
```

### Configure watched folders

Edit `utils/config.py` and update `WATCHED_FOLDERS` for your machine.

Example:

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

## Running the System

### Recommended launch

```bash
conda activate SIH
cd path/to/media-intelligence-nosql
python start_debug.py
```

What happens on startup:

1. CLIP loads
2. Flask starts at `http://127.0.0.1:5000`
3. folder watcher starts
4. full scan starts automatically
5. old documents can be backfilled with face data
6. face clustering runs after scan

### CLI note

`main.py` is a CLI command entry point and expects a subcommand.

Examples:

```bash
python main.py setup
python main.py stats
python main.py search "forest trail"
python main.py compare "mountain sunset"
python main.py evaluate
```

### Standalone web dashboard

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## How to Verify Face Processing

After running the system, you can verify that backfill and clustering completed.

### Terminal indicators

Look for:

```text
Scan complete - X new images indexed, Y existing images face-processed, Z skipped.
[Scan] Face clustering done: N people, M unmatched faces.
```

### MongoDB verification commands

Documents still missing face processing:

```bash
python -c "from database.media_repository import MediaRepository; repo=MediaRepository(); print(repo.collection.count_documents({'faces': {'$exists': False}}))"
```

Documents with at least one detected face:

```bash
python -c "from database.media_repository import MediaRepository; repo=MediaRepository(); print(repo.collection.count_documents({'faces.0': {'$exists': True}}))"
```

Cluster count:

```bash
python -c "from database.media_repository import MediaRepository; repo=MediaRepository(); print(len(repo.collection.distinct('faces.cluster_id', {'faces.cluster_id': {'$nin': [None, 'noise']}})))"
```

---

## Limitations

- face clustering currently builds a full cosine-distance matrix in memory, which may become expensive for very large face collections
- clustering is batch-oriented and not incremental
- false positives are possible on tiny faces, profile faces, screenshots, or illustrations
- absolute file paths are stored, so moving files across machines requires re-indexing
- search currently uses in-memory ranking rather than a dedicated ANN vector index

---

## Possible Future Extensions

- person-name search integrated directly into the semantic search bar
- manual cluster merge / split operations in the UI
- exclude folders from face processing separately from semantic indexing
- move from in-memory search to FAISS or MongoDB Atlas Vector Search
- incremental face clustering for newly added images
- video face keyframe support

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Database | MongoDB |
| Image-text embeddings | OpenAI CLIP via HuggingFace Transformers |
| Face detection / face embeddings | InsightFace |
| Face clustering | scikit-learn DBSCAN |
| Deep learning runtime | PyTorch |
| ONNX runtime | onnxruntime-gpu |
| Image processing | OpenCV, Pillow |
| Web framework | Flask |
| Desktop UI | tkinter |
| Tray integration | pystray |
| File watching | watchdog |

---

## Summary

This project started as a local semantic image search system and was extended into a richer NoSQL media intelligence platform.

The major additions in this update are:

- face detection
- face embedding extraction
- nested face storage in MongoDB
- face clustering
- People tab in the UI
- person labeling
- backfill support for already indexed documents

This makes the project stronger both as:

- an AI-powered media retrieval system
- a NoSQL project demonstrating evolving schema, nested documents, and document-centric design
