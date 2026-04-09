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
from faces.face_detector import detect_faces


class ImageEventHandler(FileSystemEventHandler):
    """Handles new image file events from watchdog."""

    def __init__(self, embedder, on_new_image=None):
        super().__init__()
        self.embedder      = embedder
        self.repo          = MediaRepository()
        self.on_new_image  = on_new_image
        self._processing   = set()
        self._lock         = threading.Lock()

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle(event.src_path)

    def on_moved(self, event):
        """
        Handles files moved or renamed inside a watched folder.
        Removes the old path from the index and indexes the new path.
        Covers the rename case (IMG_2847.jpg -> beach.jpg) — old document
        is cleaned up so it doesn't point to a non-existent path.
        """
        if event.is_directory:
            return

        old_path = os.path.abspath(event.src_path)
        new_path = os.path.abspath(event.dest_path)

        # Remove old path regardless of whether it was an image —
        # it may have been indexed under that path before.
        deleted = self.repo.delete_by_path(old_path)
        if deleted:
            print(f"[Watcher] Removed old path from index: {os.path.basename(old_path)}")

        # Index the new path if it is an image
        if is_image(new_path):
            self._handle(new_path)

    def on_deleted(self, event):
        """
        Handles files deleted from a watched folder while the app is running.
        Removes the document from MongoDB so the index stays clean.
        """
        if event.is_directory:
            return

        fpath = os.path.abspath(event.src_path)

        if not is_image(fpath):
            return

        deleted = self.repo.delete_by_path(fpath)
        if deleted:
            print(f"[Watcher] Removed deleted file from index: {os.path.basename(fpath)}")

    # ── Core ingestion handler ─────────────────────────────────────────────────

    def _handle(self, fpath: str):
        fpath = os.path.abspath(fpath)

        if not is_image(fpath):
            return

        with self._lock:
            if fpath in self._processing:
                return
            self._processing.add(fpath)

        try:
            # Wait for file to finish writing — check size stability
            # instead of a blind sleep so large files are handled correctly.
            if not self._wait_for_stable(fpath):
                return

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

            # Detect faces and store on the new document
            # Note: clustering is NOT triggered here (too expensive per-file).
            # The new face gets cluster_id=None until the next clustering run.
            try:
                faces = detect_faces(fpath)
                self.repo.upsert_faces(fpath, faces)
                if faces:
                    print(f"[Watcher] Detected {len(faces)} face(s): {os.path.basename(fpath)}")
            except Exception as e:
                print(f"[Watcher] Face detection failed for {os.path.basename(fpath)}: {e}")

            if self.on_new_image:
                self.on_new_image(fpath)

        finally:
            with self._lock:
                self._processing.discard(fpath)

    def _handle_moved_dest(self, fpath: str):
        """
        Same as _handle but used for the destination of a move/rename.
        Separated so on_moved can call it directly without the duplicate-check
        guard triggering if src and dest are both in the same watched folder.
        """
        self._handle(fpath)

    def _wait_for_stable(self, fpath: str, interval: float = 0.5, retries: int = 10) -> bool:
        """
        Polls the file size until it stops changing, indicating the OS has
        finished writing the file. Returns False if the file disappears or
        never stabilises within the retry window (~5 seconds default).
        Replaces the previous blind time.sleep(1.5) hack.
        """
        prev_size = -1
        for _ in range(retries):
            try:
                curr_size = os.path.getsize(fpath)
            except OSError:
                return False
            if curr_size == prev_size and curr_size > 0:
                return True
            prev_size = curr_size
            time.sleep(interval)
        return False


class FolderWatcher:
    """
    Manages watchdog Observer across all configured watched folders.
    Runs as a daemon thread — stops automatically when app exits.
    """

    def __init__(self, embedder, on_new_image=None):
        self.embedder     = embedder
        self.on_new_image = on_new_image
        self.observer     = None

    def start(self):
        handler = ImageEventHandler(self.embedder, self.on_new_image)
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
