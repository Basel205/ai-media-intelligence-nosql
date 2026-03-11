"""
Flask Web Application — Media Intelligence Dashboard
=====================================================
Run: python app.py
Opens: http://localhost:5000
"""

import os
from flask import Flask, request, jsonify, render_template, send_file, abort

app = Flask(__name__)

# ── Single shared CLIP embedder + search engines ──────────────────────────────
# Loaded once on startup to avoid loading CLIP model multiple times.
_embedder        = None
_baseline_engine = None
_css_engine      = None


def get_engines():
    global _embedder, _baseline_engine, _css_engine
    if _baseline_engine is None:
        from embeddings.clip_model import CLIPEmbedder
        from search.semantic_search import SemanticSearch
        from search.hybrid_search import CompositeSemanticSearch

        _embedder        = CLIPEmbedder()
        _baseline_engine = SemanticSearch(embedder=_embedder)
        _css_engine      = CompositeSemanticSearch(embedder=_embedder)

    return _baseline_engine, _css_engine


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def search():
    query  = request.args.get("q", "").strip()
    method = request.args.get("method", "css")
    top_k  = int(request.args.get("top_k", 12))
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
    b_results = baseline.search(query, top_k=top_k)
    c_results = css.search(query,      top_k=top_k)

    for r in b_results + c_results:
        r.pop("_breakdown", None)

    return jsonify({"query": query, "baseline": b_results, "css": c_results})


@app.route("/api/evaluate")
def evaluate():
    """Run Precision@K benchmark — reuses already-loaded engines."""
    from search.evaluation import run_evaluation
    baseline, css = get_engines()
    records = run_evaluation(
        baseline_engine=baseline,
        css_engine=css,
        k_values=(5, 10)
    )
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


@app.route("/image")
def serve_image():
    """Serve a local image file by absolute path."""
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        abort(404)
    return send_file(path)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Media Intelligence Dashboard...")
    print("Loading CLIP model (once)...")
    get_engines()   # pre-load on startup so first search is instant
    print("Ready at http://localhost:5000")
    app.run(debug=False, port=5000)