import logging

from pymongo.errors import DuplicateKeyError

from database.mongo_connection import MongoDBConnection


class MediaRepository:
    """
    All MongoDB CRUD operations for the media collection.
    Single source of truth for database access; all other modules use this.
    """

    def __init__(self):
        self.db = MongoDBConnection().get_database()
        self.collection = self.db["media"]
        self.logger = logging.getLogger(__name__)

    def insert_document(self, document: dict) -> str | None:
        """
        Insert a new document. Returns inserted _id as string, or None if
        a document with the same file_path already exists.
        """
        try:
            result = self.collection.insert_one(document)
            return str(result.inserted_id)
        except DuplicateKeyError:
            return None

    def document_exists(self, file_path: str) -> bool:
        return self.collection.find_one({"file_path": file_path}) is not None

    def needs_face_processing(self, file_path: str) -> bool:
        """
        Return True when a document exists but has not yet been through face detection.
        Documents with faces=[] count as already processed.
        """
        return (
            self.collection.find_one(
                {"file_path": file_path, "faces": {"$exists": False}},
                {"_id": 1},
            )
            is not None
        )

    def update_embeddings(self, file_id: str, clip_vector: list) -> None:
        self.collection.update_one(
            {"file_id": file_id},
            {"$set": {"embeddings.clip_image": clip_vector}},
        )

    def update_quality_flags(self, file_id: str, flags: dict) -> None:
        self.collection.update_one(
            {"file_id": file_id},
            {"$set": {"quality_flags": flags}},
        )

    def delete_by_path(self, file_path: str) -> bool:
        """
        Delete a single document by file_path.
        Returns True if a document was deleted, False if not found.
        """
        result = self.collection.delete_one({"file_path": file_path})
        return result.deleted_count > 0

    def delete_by_paths(self, file_paths: list) -> int:
        """
        Batch delete documents whose file_paths are in the given list.
        Returns the number of deleted documents.
        """
        if not file_paths:
            return 0
        result = self.collection.delete_many({"file_path": {"$in": file_paths}})
        return result.deleted_count

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

    def filter_by_metadata(self, mongo_filter: dict) -> list:
        """
        Flexible MongoDB query used by hybrid search to pre-filter
        by metadata before vector ranking.
        """
        return list(self.collection.find(mongo_filter))

    def upsert_faces(self, file_path: str, faces: list[dict]) -> bool:
        """
        Store detected faces for a document identified by file_path.

        Replaces the entire faces array, which is safe for re-ingest.
        """
        try:
            result = self.collection.update_one(
                {"file_path": file_path},
                {"$set": {"faces": faces}},
            )
            return result.matched_count > 0
        except Exception as exc:
            self.logger.error("upsert_faces failed for %s: %s", file_path, exc)
            return False

    def get_all_faces(self) -> list[dict]:
        """
        Fetch file_path + faces for all documents with a non-empty faces array.
        """
        try:
            cursor = self.collection.find(
                {"faces": {"$exists": True, "$ne": []}},
                {"file_path": 1, "faces": 1, "_id": 0},
            )
            return list(cursor)
        except Exception as exc:
            self.logger.error("get_all_faces failed: %s", exc)
            return []

    def update_face_clusters(self, face_to_cluster: dict[str, str]) -> int:
        """
        Write cluster_ids back to individual face sub-documents after clustering.
        """
        from pymongo import UpdateOne

        if not face_to_cluster:
            return 0

        operations = []
        for face_id, cluster_id in face_to_cluster.items():
            operations.append(
                UpdateOne(
                    {"faces.face_id": face_id},
                    {"$set": {"faces.$[elem].cluster_id": cluster_id}},
                    array_filters=[{"elem.face_id": face_id}],
                )
            )

        try:
            chunk_size = 1000
            total_modified = 0
            for start in range(0, len(operations), chunk_size):
                chunk = operations[start : start + chunk_size]
                result = self.collection.bulk_write(chunk, ordered=False)
                total_modified += result.modified_count
            return total_modified
        except Exception as exc:
            self.logger.error("update_face_clusters failed: %s", exc)
            return 0

    def update_person_label(self, cluster_id: str, person_label: str) -> int:
        """
        Set person_label for all faces in a cluster.
        """
        try:
            result = self.collection.update_many(
                {"faces.cluster_id": cluster_id},
                {"$set": {"faces.$[elem].person_label": person_label}},
                array_filters=[{"elem.cluster_id": cluster_id}],
            )
            return result.modified_count
        except Exception as exc:
            self.logger.error("update_person_label failed for %s: %s", cluster_id, exc)
            return 0

    def search_by_person(self, person_label: str) -> list[dict]:
        """
        Find all images containing a named person.
        """
        try:
            cursor = self.collection.find(
                {"faces.person_label": person_label},
                {
                    "file_path": 1,
                    "file_name": 1,
                    "metadata": 1,
                    "faces": 1,
                    "_id": 0,
                },
            )
            return list(cursor)
        except Exception as exc:
            self.logger.error("search_by_person failed for %s: %s", person_label, exc)
            return []

    def get_faces_by_cluster(self, cluster_id: str) -> list[dict]:
        """
        Return all images + face data for a given cluster.
        """
        try:
            cursor = self.collection.find(
                {"faces.cluster_id": cluster_id},
                {
                    "file_path": 1,
                    "file_name": 1,
                    "metadata": 1,
                    "faces": 1,
                    "_id": 0,
                },
            )
            results = []
            for doc in cursor:
                doc["faces"] = [
                    face for face in doc.get("faces", []) if face.get("cluster_id") == cluster_id
                ]
                results.append(doc)
            return results
        except Exception as exc:
            self.logger.error("get_faces_by_cluster failed for %s: %s", cluster_id, exc)
            return []

    def remove_face_from_cluster(self, face_id: str) -> bool:
        """
        Remove a face from its cluster by clearing cluster_id and person_label.
        """
        try:
            result = self.collection.update_one(
                {"faces.face_id": face_id},
                {
                    "$set": {
                        "faces.$[elem].cluster_id": None,
                        "faces.$[elem].person_label": None,
                    }
                },
                array_filters=[{"elem.face_id": face_id}],
            )
            return result.modified_count > 0
        except Exception as exc:
            self.logger.error("remove_face_from_cluster failed for %s: %s", face_id, exc)
            return False

    def count_face_clusters(self) -> int:
        """Return the number of distinct non-noise clusters."""
        try:
            cluster_ids = self.collection.distinct(
                "faces.cluster_id",
                {"faces.cluster_id": {"$nin": [None, "noise"]}},
            )
            return len(cluster_ids)
        except Exception as exc:
            self.logger.error("count_face_clusters failed: %s", exc)
            return 0
