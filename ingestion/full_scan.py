"""
Full system scan - walks all watched folders and embeds any
images not yet in MongoDB. Runs in a background thread so the
UI stays responsive. Skips excluded dirs and small files.

Also exposes reconcile_deleted() which is called at startup to remove
documents pointing to files that were deleted while the app was not running.
"""

import os
import threading
import uuid

from database.media_repository import MediaRepository
from embeddings.clip_model import CLIPEmbedder
from faces.face_clusterer import cluster_faces
from faces.face_detector import detect_faces
from ingestion.metadata_extractor import extract_metadata
from utils.config import EXCLUDED_DIRS, MIN_IMAGE_SIZE_BYTES, WATCHED_FOLDERS
from utils.file_utils import is_image


def reconcile_deleted(repo: MediaRepository, on_progress=None) -> int:
    """
    Checks every file_path stored in MongoDB and removes documents
    where the file no longer exists on disk.
    """
    print("[Reconcile] Checking index for deleted files...")

    all_docs = list(repo.collection.find({}, {"file_path": 1, "_id": 0}))
    total = len(all_docs)

    if total == 0:
        print("[Reconcile] Index is empty, nothing to check.")
        return 0

    stale_paths = []

    for index, doc in enumerate(all_docs):
        path = doc.get("file_path", "")
        if path and not os.path.exists(path):
            stale_paths.append(path)

        if on_progress and index % 500 == 0:
            on_progress(index, total)

    removed = repo.delete_by_paths(stale_paths)

    if removed:
        print(f"[Reconcile] Removed {removed} stale document(s) from index.")
    else:
        print("[Reconcile] Index is clean - no stale documents found.")

    return removed


class FullSystemScanner:
    def __init__(self, embedder: CLIPEmbedder = None, on_progress=None, on_complete=None):
        self.embedder = embedder or CLIPEmbedder()
        self.repo = MediaRepository()
        self.on_progress = on_progress
        self.on_complete = on_complete
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def _should_skip_dir(self, dirpath: str) -> bool:
        parts = dirpath.replace("\\", "/").lower().split("/")
        return any(excluded in parts for excluded in EXCLUDED_DIRS)

    def _collect_images(self) -> list[str]:
        paths = []
        for folder in WATCHED_FOLDERS:
            if not os.path.exists(folder):
                continue

            for dirpath, dirnames, filenames in os.walk(folder):
                if self._should_skip_dir(dirpath):
                    dirnames.clear()
                    continue

                dirnames[:] = [dirname for dirname in dirnames if dirname.lower() not in EXCLUDED_DIRS]

                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    if not is_image(file_path):
                        continue
                    try:
                        if os.path.getsize(file_path) < MIN_IMAGE_SIZE_BYTES:
                            continue
                    except OSError:
                        continue
                    paths.append(file_path)

        return paths

    def run(self):
        """
        Run full scan synchronously.
        Reconciles deleted files first, then scans for new images,
        backfills faces onto existing docs that do not have them yet,
        then runs face clustering over all detected faces.
        """
        reconcile_deleted(self.repo)

        print("Starting full system scan...")
        paths = self._collect_images()
        total = len(paths)
        inserted = 0
        face_backfilled = 0
        skipped = 0

        print(f"Found {total} images across watched folders.")

        for index, file_path in enumerate(paths):
            if self._stop_event.is_set():
                print("Scan stopped.")
                break

            file_name = os.path.basename(file_path)

            if self.on_progress:
                self.on_progress(index + 1, total, file_name)

            if self.repo.document_exists(file_path):
                if self.repo.needs_face_processing(file_path):
                    try:
                        faces = detect_faces(file_path)
                        self.repo.upsert_faces(file_path, faces)
                        face_backfilled += 1
                    except Exception as exc:
                        print(f"[Scan] Face backfill failed for {file_name}, skipping: {exc}")
                else:
                    skipped += 1
                continue

            metadata = extract_metadata(file_path)
            if metadata is None:
                skipped += 1
                continue

            vector = self.embedder.embed_image(file_path)
            if vector is None:
                skipped += 1
                continue

            document = {
                "file_id": str(uuid.uuid4()),
                "file_path": file_path,
                "file_name": file_name,
                "media_type": "image",
                "metadata": metadata,
                "embeddings": {"clip_image": vector},
                "quality_flags": {},
            }
            self.repo.insert_document(document)

            try:
                faces = detect_faces(file_path)
                self.repo.upsert_faces(file_path, faces)
            except Exception as exc:
                print(f"[Scan] Face detection failed for {file_name}, skipping: {exc}")

            inserted += 1

        print(
            f"Scan complete - {inserted} new images indexed, "
            f"{face_backfilled} existing images face-processed, {skipped} skipped."
        )

        print("[Scan] Running face clustering...")
        try:
            summary = cluster_faces(self.repo)
            print(
                f"[Scan] Face clustering done: {summary['num_clusters']} people, "
                f"{summary['noise_faces']} unmatched faces."
            )
        except Exception as exc:
            print(f"[Scan] Face clustering failed: {exc}")

        processed_total = inserted + face_backfilled
        if self.on_complete:
            self.on_complete(processed_total, skipped)

        return processed_total, skipped

    def run_in_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, daemon=True, name="FullScanThread")
        thread.start()
        return thread
