from database.mongo_connection import MongoDBConnection


class SchemaSetup:
    """
    Sets up MongoDB collections and indexes.

    Vector Search index note:
    ─────────────────────────
    MongoDB Atlas vector search requires creating the index via the Atlas UI
    or Atlas Search API. If using Atlas, create a Search index on the 'media'
    collection with this definition:

        {
          "fields": [
            {
              "type": "vector",
              "path": "embeddings.clip_image",
              "numDimensions": 512,
              "similarity": "cosine"
            }
          ]
        }

    For local MongoDB (non-Atlas): semantic_search.py and hybrid_search.py
    both fall back to in-memory cosine computation — no index needed.
    """

    def __init__(self):
        self.db = MongoDBConnection().get_database()

    def create_collections(self):
        if "media" not in self.db.list_collection_names():
            self.db.create_collection("media")
            print("Media collection created.")
        else:
            print("Media collection already exists.")

    def create_indexes(self):
        media = self.db["media"]
        media.create_index("file_id",           unique=True)
        media.create_index("file_path",          unique=True)
        media.create_index("metadata.blur_score")
        media.create_index("metadata.brightness")
        media.create_index("metadata.width")
        media.create_index("metadata.height")
        print("Indexes created.")

    def setup(self):
        self.create_collections()
        self.create_indexes()
        print("Schema setup complete.")
        print("NOTE: For vector search, create the Atlas Search index manually (see schema_setup.py).")
