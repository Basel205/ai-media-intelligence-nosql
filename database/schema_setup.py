from database.mongo_connection import MongoDBConnection


class SchemaSetup:

    def __init__(self):
        self.db = MongoDBConnection().get_database()

    def create_collections(self):
        """Create media collection if it does not exist."""
        if "media" not in self.db.list_collection_names():
            self.db.create_collection("media")
            print("Media collection created.")
        else:
            print("Media collection already exists.")

    def create_indexes(self):
        """Create indexes for faster querying."""
        media = self.db["media"]

        media.create_index("file_id")
        media.create_index("metadata.contains_person")
        media.create_index("metadata.blur_score")

        print("Indexes created.")

    def setup(self):
        """Run full schema setup."""
        self.create_collections()
        self.create_indexes()