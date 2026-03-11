import numpy as np

from embeddings.clip_model import CLIPEmbedder
from database.media_repository import MediaRepository


class SemanticSearch:
    """
    Baseline semantic search — pure cosine similarity between text query
    and stored CLIP image embeddings.
    Used as the comparison baseline against CSS in evaluation.py.
    """

    def __init__(self, embedder=None):
        # Accept a shared embedder to avoid loading CLIP multiple times
        self.embedder = embedder if embedder is not None else CLIPEmbedder()
        self.repo     = MediaRepository()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Returns top_k images most semantically similar to query text.
        Results sorted by cosine_similarity descending.
        """
        query_vec = self.embedder.embed_text(query)
        docs      = self.repo.get_with_embeddings()

        if not docs:
            print("No embedded documents. Run: python main.py embed")
            return []

        results = []
        for doc in docs:
            img_vec = np.array(doc["embeddings"]["clip_image"])
            sim     = float(np.dot(query_vec, img_vec))
            results.append({
                "file_id":           doc["file_id"],
                "file_name":         doc["file_name"],
                "file_path":         doc["file_path"],
                "cosine_similarity": round(sim, 4),
                "css_score":         round(sim, 4),
                "metadata":          doc.get("metadata", {}),
                "method":            "baseline"
            })

        results.sort(key=lambda x: x["cosine_similarity"], reverse=True)
        return results[:top_k]