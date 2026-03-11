import os
import uuid


SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"]


def scan_media_folder(folder_path):
    """
    Scan folder and return image file records.
    """

    media_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:

            ext = os.path.splitext(file)[1].lower()

            if ext in SUPPORTED_EXTENSIONS:

                file_path = os.path.join(root, file)

                record = {
                    "file_id": str(uuid.uuid4()),
                    "file_path": file_path,
                    "file_name": file
                }

                media_files.append(record)

    return media_files