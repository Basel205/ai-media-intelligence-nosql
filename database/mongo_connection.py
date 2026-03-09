from pymongo import MongoClient


class MongoDBConnection:
    def __init__(self, uri="mongodb://localhost:27017", db_name="media_intelligence"):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None

    def connect(self):
        """Establish connection to MongoDB."""
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        print(f"Connected to MongoDB database: {self.db_name}")

    def get_database(self):
        """Return database instance."""
        if self.db is None:
            self.connect()
        return self.db