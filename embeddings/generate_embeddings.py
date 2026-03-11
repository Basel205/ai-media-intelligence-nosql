from tqdm import tqdm

from embeddings.clip_model import CLIPEmbedder
from database.media_repository import MediaRepository


class EmbeddingGenerator:
    """
    Batch-generates CLIP embeddings for all documents that don't have one yet.
    Safe to re-run — skips already-embedded documents.
    """

    def __init__(self):
        self.embedder = CLIPEmbedder()
        self.repo     = MediaRepository()

    def run(self):
        docs = self.repo.get_without_embeddings()
        print(f"Generating embeddings for {len(docs)} documents...")

        success = 0
        failed  = 0

        for doc in tqdm(docs, desc="Embedding"):
            vector = self.embedder.embed_image(doc["file_path"])
            if vector is None:
                failed += 1
                continue
            self.repo.update_embeddings(doc["file_id"], vector)
            success += 1

        print(f"Done — {success} embedded, {failed} failed.")


if __name__ == "__main__":
    EmbeddingGenerator().run()
