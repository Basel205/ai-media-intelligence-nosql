"""
faces/face_detector.py

Detects faces in an image and returns embeddings + bounding boxes.
Uses InsightFace buffalo_l model (RetinaFace detector + ArcFace embedder).

Each returned face dict is ready to be stored as a sub-document in MongoDB.
"""

import uuid
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# InsightFace app is expensive to initialise — load once at module level.
_app = None


def _get_app():
    """Lazy-load the InsightFace FaceAnalysis app (thread-safe after first call)."""
    global _app
    if _app is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis

            # buffalo_l = RetinaFace detector + ArcFace R100 embedder (512-dim)
            # det_size must be a multiple of 32; 640x640 is the recommended default.
            _app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            _app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace FaceAnalysis loaded (buffalo_l).")
        except Exception as e:
            logger.error(f"Failed to load InsightFace: {e}")
            raise
    return _app


def detect_faces(image_path: str) -> list[dict]:
    """
    Run face detection + embedding extraction on a single image.

    Args:
        image_path: Absolute path to the image file.

    Returns:
        List of face dicts. Empty list if no faces found or image unreadable.
        Each dict:
        {
            "face_id":      str   — uuid4 string, unique per face
            "bbox":         [x1, y1, x2, y2]  — pixel coords, ints
            "embedding":    list[float]  — 512-dim ArcFace embedding (normalised)
            "det_score":    float  — RetinaFace detection confidence (0–1)
            "cluster_id":   None   — filled later by face_clusterer
            "person_label": None   — filled later by user via UI
        }
    """
    import cv2

    path = Path(image_path)
    if not path.exists():
        logger.warning(f"Image not found, skipping face detection: {image_path}")
        return []

    img = cv2.imread(str(path))
    if img is None:
        logger.warning(f"cv2 could not read image: {image_path}")
        return []

    try:
        app = _get_app()
        faces = app.get(img)
    except Exception as e:
        logger.error(f"InsightFace inference failed on {image_path}: {e}")
        return []

    if not faces:
        return []

    results = []
    for face in faces:
        # bbox comes as [x1, y1, x2, y2] floats — convert to ints
        bbox = [int(v) for v in face.bbox.tolist()]

        # ArcFace embedding: 512-dim float32, already L2-normalised by InsightFace
        embedding = face.normed_embedding.tolist()

        results.append(
            {
                "face_id": str(uuid.uuid4()),
                "bbox": bbox,
                "embedding": embedding,
                "det_score": float(face.det_score),
                "cluster_id": None,
                "person_label": None,
            }
        )

    logger.debug(f"Detected {len(results)} face(s) in {path.name}")
    return results


def detect_faces_batch(image_paths: list[str]) -> dict[str, list[dict]]:
    """
    Run face detection on a list of images.

    Returns:
        Dict mapping file_path -> list of face dicts.
        Images with no faces map to an empty list.
        Images that fail are logged and excluded from the result.
    """
    results = {}
    for path in image_paths:
        try:
            results[path] = detect_faces(path)
        except Exception as e:
            logger.error(f"Unexpected error detecting faces in {path}: {e}")
            results[path] = []
    return results