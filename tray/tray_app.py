"""
System Tray Application
=======================
Architecture:
  Main thread   -> tkinter search window (required by tkinter)
  Thread 2      -> pystray tray icon
  Thread 3      -> Flask API
  Thread 4      -> Folder watcher
  Thread 5      -> Full scan on startup
"""

import threading
import webbrowser

import pystray
from PIL import Image, ImageDraw
from pystray import MenuItem as Item


def _make_icon_image():
    """Generate tray icon programmatically."""
    img = Image.new("RGB", (64, 64), color="#0f1117")
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, 44, 44], outline="#6366f1", width=5)
    draw.line([38, 38, 58, 58], fill="#6366f1", width=5)
    return img


class TrayApp:
    def __init__(self):
        self.embedder = None
        self.watcher = None
        self.scanner = None
        self.search_window = None
        self.icon = None
        self._scanning = False

    def start(self):
        print("[App] Starting Media Intelligence...")
        self._load_clip()
        self._start_flask()
        self._start_watcher()
        self._build_search_window()
        self._start_tray_thread()
        self._start_scan(auto=True)
        print("[App] Running search window on main thread...")
        self.search_window.run()

    def _load_clip(self):
        print("[App] Loading CLIP model on GPU...")
        from embeddings.clip_model import CLIPEmbedder

        self.embedder = CLIPEmbedder()
        print("[App] CLIP ready.")

    def _start_flask(self):
        from utils.config import FLASK_HOST, FLASK_PORT

        def run():
            from app import app, init_engines

            init_engines(embedder=self.embedder)
            app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)

        threading.Thread(target=run, daemon=True, name="Flask").start()
        print("[App] Flask API starting at http://127.0.0.1:5000")

    def _start_watcher(self):
        from watcher.folder_watcher import FolderWatcher

        self.watcher = FolderWatcher(embedder=self.embedder, on_new_image=self._on_new_image)
        self.watcher.start()

    def _build_search_window(self):
        from database.media_repository import MediaRepository
        from ui.search_window import SearchWindow

        def search_fn(query, method, top_k):
            if method == "css":
                from search.hybrid_search import CompositeSemanticSearch

                engine = CompositeSemanticSearch(embedder=self.embedder)
            else:
                from search.semantic_search import SemanticSearch

                engine = SemanticSearch(embedder=self.embedder)
            return engine.search(query, top_k=top_k)

        def stats_fn():
            repo = MediaRepository()
            return {"total": repo.count(), "embedded": repo.count_embedded()}

        self.search_window = SearchWindow(search_fn=search_fn, stats_fn=stats_fn)
        self.search_window.build()
        self.search_window.hide()

    def _start_tray_thread(self):
        threading.Thread(target=self._run_tray, daemon=True, name="Tray").start()

    def _start_scan(self, auto=False):
        if self._scanning:
            print("[App] Scan already running.")
            return

        from ingestion.full_scan import FullSystemScanner

        self.scanner = FullSystemScanner(embedder=self.embedder, on_complete=self._on_scan_complete)
        self._scanning = True
        self.scanner.run_in_background()

        if auto:
            print("[App] Background scan started automatically.")

    def _run_tray(self):
        menu = pystray.Menu(
            Item("Open Search", self._open_search, default=True),
            Item("Scan Now", self._scan_now),
            Item("Recluster Faces", self._recluster_faces),
            Item("Open in Browser", self._open_browser),
            pystray.Menu.SEPARATOR,
            Item("Stats", self._show_stats),
            pystray.Menu.SEPARATOR,
            Item("Quit", self._quit),
        )
        self.icon = pystray.Icon(
            name="MediaIntelligence",
            icon=_make_icon_image(),
            title="Media Intelligence",
            menu=menu,
        )
        self.icon.run()

    def _open_search(self, icon=None, item=None):
        if self.search_window and self.search_window.root:
            self.search_window.root.after(0, self.search_window.show)

    def _scan_now(self, icon=None, item=None):
        self._start_scan(auto=False)
        self._notify("Scanning", "Background scan started.")

    def _recluster_faces(self, icon=None, item=None):
        def run():
            from database.media_repository import MediaRepository
            from faces.face_clusterer import cluster_faces

            try:
                summary = cluster_faces(MediaRepository())
                self._notify(
                    "Face Clustering",
                    f"{summary['num_clusters']} clusters, {summary['noise_faces']} noise faces.",
                )
            except Exception as exc:
                self._notify("Face Clustering", f"Failed: {exc}")

        threading.Thread(target=run, daemon=True, name="FaceClustering").start()
        self._notify("Face Clustering", "Reclustering started in the background.")

    def _open_browser(self, icon=None, item=None):
        webbrowser.open("http://127.0.0.1:5000")

    def _show_stats(self, icon=None, item=None):
        from database.media_repository import MediaRepository

        repo = MediaRepository()
        total = repo.count()
        embedded = repo.count_embedded()
        scan_state = " (scanning...)" if self._scanning else ""
        self._notify("Stats", f"Total: {total} | Indexed: {embedded}{scan_state}")

    def _quit(self, icon=None, item=None):
        print("[App] Shutting down...")
        if self.watcher:
            self.watcher.stop()
        if self.scanner:
            self.scanner.stop()
        if self.icon:
            self.icon.stop()
        if self.search_window and self.search_window.root:
            self.search_window.root.after(0, self.search_window.root.destroy)

    def _on_new_image(self, fpath):
        import os

        self._notify("New Image Indexed", os.path.basename(fpath))

    def _on_scan_complete(self, inserted, skipped):
        self._scanning = False
        self._notify("Scan Complete", f"{inserted} new images indexed, {skipped} skipped.")

    def _notify(self, title, message):
        if self.icon:
            try:
                self.icon.notify(message, title)
            except Exception:
                print(f"[{title}] {message}")
