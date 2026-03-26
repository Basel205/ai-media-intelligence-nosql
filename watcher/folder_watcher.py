"""
Folder Watcher — monitors watched folders for new images and
automatically embeds them into MongoDB as they arrive.
Uses watchdog's Observer which is already in your SIH env.
"""

import os
import uuid
import time
import threading

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from utils.config import WATCHED_FOLDERS, MIN_IMAGE_SIZE_BYTES
from utils.file_utils import is_image
from ingestion.metadata_extractor import extract_metadata
from database.media_repository import MediaRepository


class ImageEventHandler(FileSystemEventHandler):
    """Handles new image file events from watchdog."""

    def __init__(self, embedder, on_new_image=None):
        super().__init__()
        self.embedder      = embedder
        self.repo          = MediaRepository()
        self.on_new_image  = on_new_image   # optional UI callback
        self._processing   = set()          # avoid double-processing
        self._lock         = threading.Lock()

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle(event.src_path)

    def on_moved(self, event):
        """Handles files moved/renamed into a watched folder."""
        if event.is_directory:
            return
        self._handle(event.dest_path)

    def _handle(self, fpath: str):
        fpath = os.path.abspath(fpath)

        if not is_image(fpath):
            return

        with self._lock:
            if fpath in self._processing:
                return
            self._processing.add(fpath)

        try:
            # Wait briefly — file may still be being written
            time.sleep(1.5)

            if not os.path.exists(fpath):
                return

            try:
                if os.path.getsize(fpath) < MIN_IMAGE_SIZE_BYTES:
                    return
            except OSError:
                return

            if self.repo.document_exists(fpath):
                return

            metadata = extract_metadata(fpath)
            if metadata is None:
                return

            vector = self.embedder.embed_image(fpath)
            if vector is None:
                return

            document = {
                "file_id":    str(uuid.uuid4()),
                "file_path":  fpath,
                "file_name":  os.path.basename(fpath),
                "media_type": "image",
                "metadata":   metadata,
                "embeddings": {"clip_image": vector},
                "quality_flags": {}
            }
            self.repo.insert_document(document)
            print(f"[Watcher] Indexed: {os.path.basename(fpath)}")

            if self.on_new_image:
                self.on_new_image(fpath)

        finally:
            with self._lock:
                self._processing.discard(fpath)


class FolderWatcher:
    """
    Manages watchdog Observer across all configured watched folders.
    Runs as a daemon thread — stops automatically when app exits.
    """

    def __init__(self, embedder, on_new_image=None):
        self.embedder     = embedder
        self.on_new_image = on_new_image
        self.observer     = None
        self._thread      = None

    def start(self):
        handler  = ImageEventHandler(self.embedder, self.on_new_image)
        self.observer = Observer()

        scheduled = 0
        for folder in WATCHED_FOLDERS:
            if os.path.exists(folder):
                self.observer.schedule(handler, folder, recursive=True)
                scheduled += 1
                print(f"[Watcher] Monitoring: {folder}")
            else:
                print(f"[Watcher] Skipping (not found): {folder}")

        if scheduled == 0:
            print("[Watcher] No valid folders to watch.")
            return

        self.observer.start()
        print(f"[Watcher] Active on {scheduled} folder(s).")

    def stop(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            print("[Watcher] Stopped.")
