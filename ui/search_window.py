"""
Desktop Search Window — tkinter-based UI.
Opens when user clicks the system tray icon.
Closing hides it (doesn't quit the app).
"""

import os
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class SearchWindow:

    BG      = "#0f1117"
    SURFACE = "#1a1d27"
    CARD    = "#21253a"
    BORDER  = "#2e3350"
    ACCENT  = "#6366f1"
    TEXT    = "#e2e8f0"
    MUTED   = "#64748b"
    THUMB_W = 200
    THUMB_H = 150
    COLS    = 4

    def __init__(self, search_fn, stats_fn):
        self.search_fn = search_fn
        self.stats_fn  = stats_fn
        self.root      = None
        self._thumbs   = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self):
        self.root = tk.Tk()
        self.root.title("Media Intelligence — Photo Search")
        self.root.geometry("1100x750")
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self._build_header()
        self._build_search_bar()
        self._build_results_area()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=self.SURFACE, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🔍  Media Intelligence",
                 bg=self.SURFACE, fg=self.TEXT,
                 font=("Segoe UI", 14, "bold")).pack(side="left", padx=16)
        self.stats_label = tk.Label(hdr, text="",
                                    bg=self.SURFACE, fg=self.MUTED,
                                    font=("Segoe UI", 9))
        self.stats_label.pack(side="right", padx=16)

    def _build_search_bar(self):
        bar = tk.Frame(self.root, bg=self.SURFACE, pady=12)
        bar.pack(fill="x", padx=16, pady=(8, 0))

        self.query_var = tk.StringVar()
        entry = tk.Entry(bar, textvariable=self.query_var,
                         bg=self.CARD, fg=self.TEXT,
                         insertbackground=self.TEXT,
                         relief="flat", font=("Segoe UI", 12),
                         highlightthickness=1,
                         highlightbackground=self.BORDER,
                         highlightcolor=self.ACCENT)
        entry.pack(side="left", fill="x", expand=True, ipady=8, padx=(0, 8))
        entry.bind("<Return>", lambda e: self._do_search())

        tk.Button(bar, text="Search",
                  bg=self.ACCENT, fg="white",
                  relief="flat", font=("Segoe UI", 10, "bold"),
                  padx=16, pady=6, cursor="hand2",
                  command=self._do_search).pack(side="left", padx=(0, 8))

        self.method_var = tk.StringVar(value="css")
        toggle = tk.Frame(bar, bg=self.SURFACE)
        toggle.pack(side="left")
        for label, val in [("CSS", "css"), ("Baseline", "baseline")]:
            tk.Radiobutton(toggle, text=label,
                           variable=self.method_var, value=val,
                           bg=self.SURFACE, fg=self.TEXT,
                           selectcolor=self.CARD,
                           activebackground=self.SURFACE,
                           font=("Segoe UI", 9)).pack(side="left")

        tk.Label(bar, text="Show:", bg=self.SURFACE, fg=self.MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(12, 4))
        self.topk_var = tk.IntVar(value=20)
        ttk.Combobox(bar, textvariable=self.topk_var,
                     values=[10, 20, 40, 60], width=4,
                     state="readonly").pack(side="left")

        self.status_var = tk.StringVar(value="Enter a search query above.")
        tk.Label(self.root, textvariable=self.status_var,
                 bg=self.BG, fg=self.MUTED,
                 font=("Segoe UI", 9)).pack(pady=(8, 0))

    def _build_results_area(self):
        container = tk.Frame(self.root, bg=self.BG)
        container.pack(fill="both", expand=True, padx=16, pady=8)

        canvas = tk.Canvas(container, bg=self.BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical",
                                  command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.results_frame = tk.Frame(canvas, bg=self.BG)
        self.canvas_window = canvas.create_window(
            (0, 0), window=self.results_frame, anchor="nw")

        self.results_frame.bind("<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
            lambda e: canvas.itemconfig(
                self.canvas_window, width=e.width))
        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units"))
        self.canvas = canvas

    # ── Search ────────────────────────────────────────────────────────────────

    def _do_search(self):
        query = self.query_var.get().strip()
        if not query:
            return
        self.status_var.set(f'Searching for "{query}"...')
        self._clear_results()

        def run():
            try:
                results = self.search_fn(
                    query,
                    self.method_var.get(),
                    self.topk_var.get()
                )
                # Schedule UI update back on main thread
                self.root.after(0, lambda: self._show_results(results, query))
            except Exception as e:
                self.root.after(0,
                    lambda: self.status_var.set(f"Error: {e}"))

        threading.Thread(target=run, daemon=True).start()

    def _clear_results(self):
        for w in self.results_frame.winfo_children():
            w.destroy()
        self._thumbs.clear()

    def _show_results(self, results, query):
        method = "CSS" if self.method_var.get() == "css" else "Baseline"
        self.status_var.set(
            f'{len(results)} results for "{query}" — {method}')
        if not results:
            tk.Label(self.results_frame, text="No results found.",
                     bg=self.BG, fg=self.MUTED,
                     font=("Segoe UI", 11)).grid(
                         row=0, column=0, pady=40)
            return
        for i, r in enumerate(results):
            row, col = divmod(i, self.COLS)
            self._make_card(r, row, col)

    def _make_card(self, result, row, col):
        card = tk.Frame(self.results_frame, bg=self.CARD, padx=4, pady=4)
        card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
        self.results_frame.columnconfigure(col, weight=1)

        thumb = self._load_thumbnail(result["file_path"])
        if thumb:
            lbl = tk.Label(card, image=thumb, bg=self.CARD, cursor="hand2")
            lbl.image = thumb
            lbl.pack()
            self._thumbs.append(thumb)
            lbl.bind("<Button-1>",
                lambda e, p=result["file_path"]: os.startfile(p))
        else:
            tk.Label(card, text="[No Preview]",
                     bg=self.CARD, fg=self.MUTED,
                     width=24, height=8).pack()

        fname = result["file_name"]
        if len(fname) > 22:
            fname = fname[:19] + "..."
        tk.Label(card, text=fname, bg=self.CARD, fg=self.TEXT,
                 font=("Segoe UI", 8), wraplength=190).pack(pady=(4, 0))

        score = result.get("css_score", result.get("cosine_similarity", 0))
        color = self.ACCENT if result.get("method") == "css" else self.MUTED
        tk.Label(card, text=f"Score: {score:.3f}",
                 bg=self.CARD, fg=color,
                 font=("Segoe UI", 8, "bold")).pack()

    def _load_thumbnail(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((self.THUMB_W, self.THUMB_H), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    # ── Stats (called from main thread only) ──────────────────────────────────

    def refresh_stats(self):
        """Call this from main thread to update the stats label."""
        try:
            s    = self.stats_fn()
            text = f"Total: {s.get('total', 0)}  |  Indexed: {s.get('embedded', 0)}"
            self.stats_label.config(text=text)
        except Exception:
            pass

    # ── Show / Hide ───────────────────────────────────────────────────────────

    def show(self):
        if self.root:
            self.refresh_stats()
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()

    def hide(self):
        if self.root:
            self.root.withdraw()

    def run(self):
        if self.root is None:
            self.build()
        self.root.mainloop()
