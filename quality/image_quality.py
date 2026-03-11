from database.media_repository import MediaRepository
from utils.config import MIN_BLUR_SCORE, MIN_BRIGHTNESS


def assess_quality(metadata: dict) -> dict:
    """
    Assess image quality and return quality flags.
    Called during or after ingestion.
    """
    blur       = metadata.get("blur_score", 0)
    brightness = metadata.get("brightness", 0)

    is_blurry  = blur < MIN_BLUR_SCORE
    is_dark    = brightness < MIN_BRIGHTNESS
    is_overexp = brightness > 0.85

    tier = "high"
    if is_blurry or is_dark or is_overexp:
        tier = "low"
    elif blur < MIN_BLUR_SCORE * 2:
        tier = "medium"

    return {
        "is_blurry":      is_blurry,
        "is_dark":        is_dark,
        "is_overexposed": is_overexp,
        "quality_tier":   tier
    }


def tag_all_documents():
    """Retroactively add quality_flags to all documents in the collection."""
    repo = MediaRepository()
    docs = repo.get_all()
    print(f"Tagging quality for {len(docs)} documents...")
    for doc in docs:
        flags = assess_quality(doc.get("metadata", {}))
        repo.update_quality_flags(doc["file_id"], flags)
    print("Quality tagging complete.")
