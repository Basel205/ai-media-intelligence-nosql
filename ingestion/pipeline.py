from tqdm import tqdm

from ingestion.scanner import scan_media_folder
from ingestion.metadata_extractor import extract_metadata
from database.media_repository import MediaRepository


class MediaIngestionPipeline:

    def __init__(self, media_folder: str):
        self.media_folder = media_folder
        self.repo = MediaRepository()

    def run(self):
        files = scan_media_folder(self.media_folder)
        print(f"Found {len(files)} media files in: {self.media_folder}")

        inserted = 0
        skipped  = 0

        for file in tqdm(files, desc="Ingesting"):

            # Skip already-ingested files (safe to re-run)
            if self.repo.document_exists(file["file_path"]):
                skipped += 1
                continue

            metadata = extract_metadata(file["file_path"])
            if metadata is None:
                skipped += 1
                continue

            document = {
                "file_id":    file["file_id"],
                "file_path":  file["file_path"],
                "file_name":  file["file_name"],
                "media_type": "image",
                "metadata":   metadata,
                "embeddings": {},
                "quality_flags": {}
            }

            self.repo.insert_document(document)
            inserted += 1

        print(f"Done — {inserted} inserted, {skipped} skipped.")
