import numpy as np
from database.media_repository import MediaRepository
from utils.config import DUPLICATE_THRESHOLD


class DuplicateDetector:
    """
    Near-duplicate detection using cosine similarity of CLIP embeddings.
    Two images are duplicates if similarity >= threshold (default 0.97).
    Catches both exact copies and near-identical images (different resolutions, crops).
    """

    def __init__(self, threshold: float = DUPLICATE_THRESHOLD):
        self.threshold = threshold
        self.repo      = MediaRepository()

    def find_duplicates(self) -> list[dict]:
        docs = self.repo.get_with_embeddings()
        if len(docs) < 2:
            print("Need at least 2 embedded images.")
            return []

        items = [{"file_id": d["file_id"], "file_name": d["file_name"],
                  "file_path": d["file_path"],
                  "vec": np.array(d["embeddings"]["clip_image"])} for d in docs]

        pairs = []
        n = len(items)
        print(f"Comparing {n} images (threshold={self.threshold})...")

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(items[i]["vec"], items[j]["vec"]))
                if sim >= self.threshold:
                    pairs.append({
                        "image_a":    items[i]["file_name"],
                        "image_b":    items[j]["file_name"],
                        "path_a":     items[i]["file_path"],
                        "path_b":     items[j]["file_path"],
                        "similarity": round(sim, 4)
                    })

        pairs.sort(key=lambda x: x["similarity"], reverse=True)
        print(f"Found {len(pairs)} duplicate pair(s).")
        return pairs

    def report(self):
        pairs = self.find_duplicates()
        if not pairs:
            print("No duplicates found.")
            return
        print(f"\n{'Image A':<35} {'Image B':<35} {'Sim':>6}")
        print("─" * 78)
        for p in pairs:
            print(f"{p['image_a']:<35} {p['image_b']:<35} {p['similarity']:>6.4f}")
