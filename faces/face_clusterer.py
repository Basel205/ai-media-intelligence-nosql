"""
faces/face_clusterer.py

Clusters all face embeddings stored in MongoDB using DBSCAN.

Why DBSCAN (not K-Means):
  - No need to specify number of people in advance
  - Handles noise / one-off faces that don't belong to any cluster
  - Works well on cosine-space embeddings when we use precomputed distance matrix

After clustering, each face document in MongoDB gets its cluster_id updated.
Noise faces (cluster = -1 from DBSCAN) get cluster_id = "noise".

Run this:
  - After a full scan completes (called from full_scan.py)
  - On demand from the tray menu ("Re-cluster faces")
  - After new images are ingested in bulk (optional, heavy operation)

Do NOT run on every single image insert — too slow. Batch it.
"""

import logging
import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def cluster_faces(repo, eps: float = 0.4, min_samples: int = 2) -> dict:
    """
    Fetch all face embeddings from MongoDB, run DBSCAN, write cluster_ids back.

    Args:
        repo:        MediaRepository instance (for DB access)
        eps:         DBSCAN epsilon — max cosine distance between neighbours.
                     0.4 works well for ArcFace 512-dim embeddings.
                     Lower = stricter (fewer, purer clusters).
                     Raise to 0.5–0.6 if the same person splits into many clusters.
        min_samples: Minimum faces to form a cluster (2 = any pair counts).
                     Raise to 3–4 to suppress noisy singletons.

    Returns:
        Summary dict:
        {
            "total_faces":    int,
            "num_clusters":   int,   # excludes noise
            "noise_faces":    int,
            "updated_docs":   int,
        }
    """
    logger.info("Starting face clustering...")

    # ------------------------------------------------------------------ #
    # 1. Fetch all face embeddings from MongoDB                           #
    # ------------------------------------------------------------------ #
    # Pull only the fields we need — avoid loading clip_image embeddings.
    # Returns list of dicts: {file_path, faces: [{face_id, embedding, ...}]}
    docs_with_faces = repo.get_all_faces()  # defined in media_repository.py

    if not docs_with_faces:
        logger.info("No faces found in database. Skipping clustering.")
        return {"total_faces": 0, "num_clusters": 0, "noise_faces": 0, "updated_docs": 0}

    # Flatten into parallel arrays
    face_ids = []       # str
    file_paths = []     # str
    embeddings = []     # list[list[float]]

    for doc in docs_with_faces:
        for face in doc.get("faces", []):
            face_ids.append(face["face_id"])
            file_paths.append(doc["file_path"])
            embeddings.append(face["embedding"])

    total_faces = len(face_ids)
    if total_faces == 0:
        logger.info("All face arrays are empty. Skipping clustering.")
        return {"total_faces": 0, "num_clusters": 0, "noise_faces": 0, "updated_docs": 0}

    logger.info(f"Clustering {total_faces} faces from {len(docs_with_faces)} images...")

    # ------------------------------------------------------------------ #
    # 2. Build distance matrix and run DBSCAN                            #
    # ------------------------------------------------------------------ #
    X = np.array(embeddings, dtype=np.float32)  # shape (N, 512)

    # ArcFace embeddings are L2-normalised, so cosine distance = 1 - dot product.
    # We compute the full (N, N) cosine distance matrix for DBSCAN.
    # For N > 50k this can be large (~10GB at float32) — fine for typical photo
    # libraries; add batching here if needed for very large collections.
    dot_products = X @ X.T                          # (N, N), values in [-1, 1]
    cosine_distances = np.clip(1.0 - dot_products, 0.0, 2.0)  # (N, N)

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
        n_jobs=-1,
    )
    labels = db.fit_predict(cosine_distances)  # shape (N,), -1 = noise

    # ------------------------------------------------------------------ #
    # 3. Build cluster_id map: face_id -> cluster_id string              #
    # ------------------------------------------------------------------ #
    unique_clusters = set(labels) - {-1}
    num_clusters = len(unique_clusters)
    noise_count = int(np.sum(labels == -1))

    # Give each real cluster a stable string ID: "cluster_0000", "cluster_0001", ...
    face_to_cluster: dict[str, str] = {}
    for face_id, label in zip(face_ids, labels):
        if label == -1:
            face_to_cluster[face_id] = "noise"
        else:
            face_to_cluster[face_id] = f"cluster_{label:04d}"

    # ------------------------------------------------------------------ #
    # 4. Write cluster_ids back to MongoDB                               #
    # ------------------------------------------------------------------ #
    updated_docs = repo.update_face_clusters(face_to_cluster)

    logger.info(
        f"Clustering complete. "
        f"{num_clusters} clusters, {noise_count} noise faces, "
        f"{updated_docs} documents updated."
    )

    return {
        "total_faces": total_faces,
        "num_clusters": num_clusters,
        "noise_faces": noise_count,
        "updated_docs": updated_docs,
    }


def get_cluster_representatives(repo) -> list[dict]:
    """
    For the People tab UI: return one representative face per cluster.

    Picks the face with the highest det_score in each cluster as the thumbnail.

    Returns:
        List of dicts:
        [
            {
                "cluster_id":   "cluster_0001",
                "person_label": "Sarah" or None,
                "representative": {
                    "file_path": "...",
                    "bbox":      [x1, y1, x2, y2],
                    "face_id":   "...",
                    "det_score": 0.98,
                }
            },
            ...
        ]
        Sorted by cluster_id. Noise cluster excluded.
    """
    docs = repo.get_all_faces()

    # cluster_id -> list of (face dict, file_path)
    clusters: dict[str, list] = {}

    for doc in docs:
        for face in doc.get("faces", []):
            cid = face.get("cluster_id")
            if cid is None or cid == "noise":
                continue
            if cid not in clusters:
                clusters[cid] = []
            clusters[cid].append((face, doc["file_path"]))

    result = []
    for cid, entries in sorted(clusters.items()):
        # Pick highest confidence face as representative
        best_face, best_path = max(entries, key=lambda e: e[0].get("det_score", 0))
        result.append(
            {
                "cluster_id": cid,
                "person_label": best_face.get("person_label"),
                "face_count": len(entries),
                "representative": {
                    "file_path": best_path,
                    "bbox": best_face["bbox"],
                    "face_id": best_face["face_id"],
                    "det_score": best_face.get("det_score", 0),
                },
            }
        )

    return result