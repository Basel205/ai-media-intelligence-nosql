"""
Full system scan — walks all watched folders and embeds any
images not yet in MongoDB. Runs in a background thread so the
UI stays responsive. Skips excluded dirs and small files (icons/thumbs).
"""

import os
import uuid
import threading
from tqdm import tqdm

from utils.config import WATCHED_FOLDERS, EXCLUDED_DIRS, MIN_IMAGE_SIZE_BYTES
from utils.file_utils import is_image
from ingestion.metadata_extractor import extract_metadata
from database.media_repository import MediaRepository
from embeddings.clip_model import CLIPEmbedder


class FullSystemScanner:

    def __init__(self, embedder: CLIPEmbedder = None, on_progress=None, on_complete=None):
        """
        Args:
            embedder:    Shared CLIPEmbedder instance (avoids loading model twice)
            on_progress: Optional callback(current, total, filename) for UI updates
            on_complete: Optional callback(inserted, skipped) called when done
        """
        self.embedder    = embedder or CLIPEmbedder()
        self.repo        = MediaRepository()
        self.on_progress = on_progress
        self.on_complete = on_complete
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _should_skip_dir(self, dirpath: str) -> bool:
        parts = dirpath.replace("\\", "/").lower().split("/")
        return any(excl in parts for excl in EXCLUDED_DIRS)

    def _collect_images(self) -> list[str]:
        """Walk all watched folders and collect image paths."""
        paths = []
        for folder in WATCHED_FOLDERS:
            if not os.path.exists(folder):
                continue
            for dirpath, dirnames, filenames in os.walk(folder):
                if self._should_skip_dir(dirpath):
                    dirnames.clear()
                    continue
                # Prune excluded subdirs in-place (stops os.walk descending)
                dirnames[:] = [
                    d for d in dirnames
                    if d.lower() not in EXCLUDED_DIRS
                ]
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    if not is_image(fpath):
                        continue
                    try:
                        if os.path.getsize(fpath) < MIN_IMAGE_SIZE_BYTES:
                            continue
                    except OSError:
                        continue
                    paths.append(fpath)
        return paths

    def run(self):
        """Run full scan synchronously."""
        print("Starting full system scan...")
        paths    = self._collect_images()
        total    = len(paths)
        inserted = 0
        skipped  = 0

        print(f"Found {total} images across watched folders.")

        for i, fpath in enumerate(paths):
            if self._stop_event.is_set():
                print("Scan stopped.")
                break

            fname = os.path.basename(fpath)

            if self.on_progress:
                self.on_progress(i + 1, total, fname)

            # Skip already processed
            if self.repo.document_exists(fpath):
                skipped += 1
                continue

            metadata = extract_metadata(fpath)
            if metadata is None:
                skipped += 1
                continue

            vector = self.embedder.embed_image(fpath)
            if vector is None:
                skipped += 1
                continue

            document = {
                "file_id":    str(uuid.uuid4()),
                "file_path":  fpath,
                "file_name":  fname,
                "media_type": "image",
                "metadata":   metadata,
                "embeddings": {"clip_image": vector},
                "quality_flags": {}
            }
            self.repo.insert_document(document)
            inserted += 1

        print(f"Scan complete — {inserted} new images indexed, {skipped} skipped.")

        if self.on_complete:
            self.on_complete(inserted, skipped)

        return inserted, skipped

    def run_in_background(self) -> threading.Thread:
        """Launch scan in a daemon thread. Returns the thread."""
        t = threading.Thread(target=self.run, daemon=True, name="FullScanThread")
        t.start()
        return t
