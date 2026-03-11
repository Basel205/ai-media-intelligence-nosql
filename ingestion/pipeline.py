from tqdm import tqdm

from ingestion.scanner import scan_media_folder
from ingestion.metadata_extractor import extract_metadata
from database.mongo_connection import MongoDBConnection


class MediaIngestionPipeline:

    def __init__(self, media_folder):
        self.media_folder = media_folder
        self.db = MongoDBConnection().get_database()
        self.collection = self.db["media"]

    def run(self):

        files = scan_media_folder(self.media_folder)

        print(f"Found {len(files)} media files")

        for file in tqdm(files):

            metadata = extract_metadata(file["file_path"])

            if metadata is None:
                continue

            document = {
                "file_id": file["file_id"],
                "file_path": file["file_path"],
                "file_name": file["file_name"],
                "media_type": "image",
                "metadata": metadata,
                "embeddings": {},
                "quality_flags": {}
            }

            self.collection.insert_one(document)