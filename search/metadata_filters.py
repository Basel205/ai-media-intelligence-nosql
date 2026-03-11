"""
Standalone metadata filter helpers.
Used by hybrid_search.py and the Flask API.
"""

from database.media_repository import MediaRepository


def build_mongo_filter(
    min_width:      int   = None,
    min_height:     int   = None,
    min_blur:       float = None,
    min_brightness: float = None,
    max_brightness: float = None,
) -> dict:
    """Build a MongoDB query filter dict from optional quality constraints."""
    f = {"embeddings.clip_image": {"$exists": True}}

    if min_width      is not None: f["metadata.width"]       = {"$gte": min_width}
    if min_height     is not None: f["metadata.height"]      = {"$gte": min_height}
    if min_brightness is not None:
        f.setdefault("metadata.brightness", {})["$gte"] = min_brightness
    if max_brightness is not None:
        f.setdefault("metadata.brightness", {})["$lte"] = max_brightness
    if min_blur       is not None: f["metadata.blur_score"]  = {"$gte": min_blur}

    return f


def get_quality_stats() -> dict:
    """Return aggregate quality statistics across the collection."""
    repo = MediaRepository()
    docs = repo.get_with_embeddings()
    if not docs:
        return {}

    blurs        = [d["metadata"]["blur_score"]  for d in docs if "blur_score"  in d.get("metadata", {})]
    brightnesses = [d["metadata"]["brightness"]  for d in docs if "brightness"  in d.get("metadata", {})]

    def stats(values):
        if not values:
            return {}
        return {
            "min":  round(min(values), 3),
            "max":  round(max(values), 3),
            "mean": round(sum(values) / len(values), 3),
        }

    return {
        "total":      len(docs),
        "blur":       stats(blurs),
        "brightness": stats(brightnesses),
    }