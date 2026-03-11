import os
import uuid
from utils.file_utils import get_all_image_paths


def scan_media_folder(folder_path: str) -> list[dict]:
    """
    Recursively scan a folder and return a list of file descriptor dicts.
    Each dict contains file_id, file_path, and file_name.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Media folder not found: {folder_path}")

    paths = get_all_image_paths(folder_path)

    files = []
    for path in paths:
        files.append({
            "file_id":   str(uuid.uuid4()),
            "file_path": os.path.abspath(path),
            "file_name": os.path.basename(path)
        })

    return files
