"""
Flask Web Application — Media Intelligence API
===============================================
Can be used standalone (python app.py) or launched by tray_app.py.
When launched by tray_app, it receives a shared CLIPEmbedder instance
so CLIP is only loaded once across the entire application.
"""

import os
from flask import Flask, request, jsonify, render_template, send_file, abort

app = Flask(__name__)

# Shared engines — set by tray_app or lazy-loaded on first request
_embedder        = None
_baseline_engine = None
_css_engine      = None


def init_engines(embedder=None):
    """
    Called by tray_app.py to inject a pre-loaded embedder.
    If not called, engines are lazy-loaded on first request.
    """
    global _embedder, _baseline_engine, _css_engine
    from search.semantic_search import SemanticSearch
    from search.hybrid_search import CompositeSemanticSearch

    if embedder:
        _embedder = embedder
    else:
        from embeddings.clip_model import CLIPEmbedder
        _embedder = CLIPEmbedder()

    _baseline_engine = SemanticSearch(embedder=_embedder)
    _css_engine      = CompositeSemanticSearch(embedder=_embedder)


def get_engines():
    global _baseline_engine, _css_engine
    if _baseline_engine is None:
        init_engines()
    return _baseline_engine, _css_engine


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def search():
    query          = request.args.get("q", "").strip()
    method         = request.args.get("method", "css")
    top_k          = int(request.args.get("top_k", 20))
    min_blur       = request.args.get("min_blur",       type=float)
    min_brightness = request.args.get("min_brightness", type=float)

    if not query:
        return jsonify({"error": "Empty query"}), 400

    baseline, css = get_engines()
    if method == "baseline":
        results = baseline.search(query, top_k=top_k)
    else:
        results = css.search(query, top_k=top_k,
                             min_blur=min_blur,
                             min_brightness=min_brightness)

    for r in results:
        r.pop("_breakdown", None)

    return jsonify({"query": query, "method": method, "results": results})


@app.route("/api/compare")
def compare():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 6))
    if not query:
        return jsonify({"error": "Empty query"}), 400

    baseline, css = get_engines()
    b = baseline.search(query, top_k=top_k)
    c = css.search(query,      top_k=top_k)

    for r in b + c:
        r.pop("_breakdown", None)

    return jsonify({"query": query, "baseline": b, "css": c})


@app.route("/api/evaluate")
def evaluate():
    from search.evaluation import run_evaluation
    baseline, css = get_engines()
    records = run_evaluation(baseline_engine=baseline,
                             css_engine=css,
                             k_values=(5, 10))
    return jsonify({"records": records})


@app.route("/api/stats")
def stats():
    from database.media_repository import MediaRepository
    from search.metadata_filters import get_quality_stats
    repo = MediaRepository()
    return jsonify({
        "total":    repo.count(),
        "embedded": repo.count_embedded(),
        "quality":  get_quality_stats(),
    })


@app.route("/api/scan", methods=["POST"])
def trigger_scan():
    """Trigger a manual scan via API."""
    return jsonify({"status": "Scan triggered via CLI. Use tray for full scan."}), 200


@app.route("/image")
def serve_image():
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        abort(404)
    return send_file(path)


# ── Standalone run ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting standalone Flask server...")
    init_engines()
    print("Ready at http://localhost:5000")
    app.run(debug=False, port=5000, host="127.0.0.1")
