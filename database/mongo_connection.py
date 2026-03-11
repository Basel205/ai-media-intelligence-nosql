from pymongo import MongoClient
from utils.config import MONGO_URI, DATABASE_NAME


class MongoDBConnection:
    def __init__(self, uri=MONGO_URI, db_name=DATABASE_NAME):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None

    def connect(self):
        """Establish connection to MongoDB with validation."""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # Forces actual connection check
            self.db = self.client[self.db_name]
            print(f"Connected to MongoDB: {self.db_name}")
        except Exception as e:
            raise ConnectionError(f"MongoDB connection failed: {e}")

    def get_database(self):
        if self.db is None:
            self.connect()
        return self.db

    def close(self):
        if self.client:
            self.client.close()
