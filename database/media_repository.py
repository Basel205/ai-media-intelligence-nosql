from database.mongo_connection import MongoDBConnection


class MediaRepository:
    """
    All MongoDB CRUD operations for the media collection.
    Single source of truth for database access — all other modules use this.
    """

    def __init__(self):
        self.db = MongoDBConnection().get_database()
        self.collection = self.db["media"]

    # ── Write ─────────────────────────────────────────────────────────────────

    def insert_document(self, document: dict) -> str:
        result = self.collection.insert_one(document)
        return str(result.inserted_id)

    def document_exists(self, file_path: str) -> bool:
        return self.collection.find_one({"file_path": file_path}) is not None

    def update_embeddings(self, file_id: str, clip_vector: list) -> None:
        self.collection.update_one(
            {"file_id": file_id},
            {"$set": {"embeddings.clip_image": clip_vector}}
        )

    def update_quality_flags(self, file_id: str, flags: dict) -> None:
        self.collection.update_one(
            {"file_id": file_id},
            {"$set": {"quality_flags": flags}}
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_by_file_id(self, file_id: str) -> dict | None:
        return self.collection.find_one({"file_id": file_id})

    def get_all(self) -> list:
        return list(self.collection.find({}))

    def get_without_embeddings(self) -> list:
        return list(self.collection.find({"embeddings.clip_image": {"$exists": False}}))

    def get_with_embeddings(self) -> list:
        return list(self.collection.find({"embeddings.clip_image": {"$exists": True}}))

    def count(self) -> int:
        return self.collection.count_documents({})

    def count_embedded(self) -> int:
        return self.collection.count_documents({"embeddings.clip_image": {"$exists": True}})

    # ── Filtered queries (used by hybrid search) ──────────────────────────────

    def filter_by_metadata(self, mongo_filter: dict) -> list:
        """
        Flexible MongoDB query used by hybrid search to pre-filter
        by metadata before vector ranking.

        Example:
            {"metadata.blur_score": {"$gte": 100}, "metadata.width": {"$gte": 1280}}
        """
        return list(self.collection.find(mongo_filter))
