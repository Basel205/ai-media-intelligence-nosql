import numpy as np

from embeddings.clip_model import CLIPEmbedder
from database.media_repository import MediaRepository
from utils.config import CSS_ALPHA, CSS_BETA, CSS_GAMMA, CSS_MAX_BLUR


class CompositeSemanticSearch:
    """
    Composite Semantic Similarity (CSS) Search
    ═══════════════════════════════════════════
    Project novelty: a custom re-ranking algorithm that extends standard
    cosine vector search by incorporating image quality metadata.

    Motivation
    ──────────
    Pure cosine search (baseline) ranks images solely by CLIP embedding
    proximity. This means a blurry, dark, low-quality image can outrank a
    sharp one if its embedding is marginally closer to the query.

    CSS fixes this by combining three signals:

        CSS(q, img) = α · cosine_sim(q_text, img_clip)   — semantic match
                    + β · sharpness(img)                  — blur quality
                    + γ · brightness(img)                 — exposure quality

        Default weights: α=0.60, β=0.25, γ=0.15  (sum = 1.0)

    Hybrid query
    ────────────
    Before vector ranking, an optional MongoDB metadata pre-filter is applied
    (min resolution, min sharpness, etc.). This is a true hybrid NoSQL query:
    structured metadata filtering + unstructured vector similarity ranking.
    """

    def __init__(self, alpha=CSS_ALPHA, beta=CSS_BETA, gamma=CSS_GAMMA, embedder=None):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "CSS weights must sum to 1.0"
        self.alpha    = alpha
        self.beta     = beta
        self.gamma    = gamma
        # Accept a shared embedder to avoid loading CLIP multiple times
        self.embedder = embedder if embedder is not None else CLIPEmbedder()
        self.repo     = MediaRepository()

    # ── Normalisation helpers ─────────────────────────────────────────────────

    def _norm_blur(self, blur: float) -> float:
        """Map Laplacian variance → [0, 1]. Higher = sharper."""
        return min(blur / CSS_MAX_BLUR, 1.0)

    def _norm_brightness(self, brightness: float) -> float:
        """
        Tent function peaking at 0.5 (ideal exposure).
        Penalises very dark (<0.15) and overexposed (>0.85) images.
        """
        return max(0.0, 1.0 - abs(brightness - 0.5) * 2.0)

    # ── Composite score ───────────────────────────────────────────────────────

    def _css(self, cosine: float, meta: dict) -> float:
        blur_n   = self._norm_blur(meta.get("blur_score", 0))
        bright_n = self._norm_brightness(meta.get("brightness", 0.5))
        return round(self.alpha * cosine + self.beta * blur_n + self.gamma * bright_n, 4)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:          str,
        top_k:          int   = 10,
        min_width:      int   = None,
        min_blur:       float = None,
        min_brightness: float = None,
    ) -> list[dict]:
        """
        Hybrid search: MongoDB metadata pre-filter → CLIP vector ranking → CSS re-rank.
        """
        query_vec = self.embedder.embed_text(query)

        # MongoDB hybrid filter
        mongo_filter = {"embeddings.clip_image": {"$exists": True}}
        if min_width      is not None: mongo_filter["metadata.width"]      = {"$gte": min_width}
        if min_blur       is not None: mongo_filter["metadata.blur_score"] = {"$gte": min_blur}
        if min_brightness is not None: mongo_filter["metadata.brightness"] = {"$gte": min_brightness}

        docs = self.repo.filter_by_metadata(mongo_filter)
        if not docs:
            print("No documents match the filters.")
            return []

        results = []
        for doc in docs:
            img_vec = np.array(doc["embeddings"]["clip_image"])
            cosine  = float(np.dot(query_vec, img_vec))
            meta    = doc.get("metadata", {})
            css     = self._css(cosine, meta)

            results.append({
                "file_id":           doc["file_id"],
                "file_name":         doc["file_name"],
                "file_path":         doc["file_path"],
                "cosine_similarity": round(cosine, 4),
                "css_score":         css,
                "metadata":          meta,
                "method":            "css",
                "_breakdown": {
                    "cosine":          round(cosine, 4),
                    "sharpness_norm":  round(self._norm_blur(meta.get("blur_score", 0)), 4),
                    "brightness_norm": round(self._norm_brightness(meta.get("brightness", 0.5)), 4),
                }
            })

        results.sort(key=lambda x: x["css_score"], reverse=True)
        return results[:top_k]