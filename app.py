"""
Flask Web Application - Media Intelligence API
==============================================
Can be used standalone (python app.py) or launched by tray_app.py.
When launched by tray_app, it receives a shared CLIPEmbedder instance
so CLIP is only loaded once across the entire application.
"""

import io
import os

from flask import Flask, abort, jsonify, render_template, request, send_file

app = Flask(__name__)

# Shared engines - set by tray_app or lazy-loaded on first request
_embedder = None
_baseline_engine = None
_css_engine = None


def init_engines(embedder=None):
    """
    Called by tray_app.py to inject a pre-loaded embedder.
    If not called, engines are lazy-loaded on first request.
    """
    global _embedder, _baseline_engine, _css_engine

    from search.hybrid_search import CompositeSemanticSearch
    from search.semantic_search import SemanticSearch

    if embedder:
        _embedder = embedder
    else:
        from embeddings.clip_model import CLIPEmbedder

        _embedder = CLIPEmbedder()

    _baseline_engine = SemanticSearch(embedder=_embedder)
    _css_engine = CompositeSemanticSearch(embedder=_embedder)


def get_engines():
    global _baseline_engine, _css_engine
    if _baseline_engine is None:
        init_engines()
    return _baseline_engine, _css_engine


def get_repo():
    from database.media_repository import MediaRepository

    return MediaRepository()


def parse_top_k(default=20):
    try:
        return max(1, int(request.args.get("top_k", default)))
    except (TypeError, ValueError):
        return default


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()
    method = request.args.get("method", "css")
    top_k = parse_top_k(20)
    min_blur = request.args.get("min_blur", type=float)
    min_brightness = request.args.get("min_brightness", type=float)

    if not query:
        return jsonify({"error": "Empty query"}), 400

    baseline, css = get_engines()
    if method == "baseline":
        results = baseline.search(query, top_k=top_k)
    else:
        results = css.search(
            query,
            top_k=top_k,
            min_blur=min_blur,
            min_brightness=min_brightness,
        )

    for result in results:
        result.pop("_breakdown", None)

    return jsonify({"query": query, "method": method, "results": results})


@app.route("/api/compare")
def compare():
    query = request.args.get("q", "").strip()
    top_k = parse_top_k(6)
    if not query:
        return jsonify({"error": "Empty query"}), 400

    baseline, css = get_engines()
    baseline_results = baseline.search(query, top_k=top_k)
    css_results = css.search(query, top_k=top_k)

    for result in baseline_results + css_results:
        result.pop("_breakdown", None)

    return jsonify({"query": query, "baseline": baseline_results, "css": css_results})


@app.route("/api/evaluate")
def evaluate():
    from search.evaluation import run_evaluation

    baseline, css = get_engines()
    records = run_evaluation(baseline_engine=baseline, css_engine=css, k_values=(5, 10))
    return jsonify({"records": records})


@app.route("/api/stats")
def stats():
    from search.metadata_filters import get_quality_stats

    repo = get_repo()
    return jsonify(
        {
            "total": repo.count(),
            "embedded": repo.count_embedded(),
            "people_clusters": repo.count_face_clusters(),
            "quality": get_quality_stats(),
        }
    )


@app.route("/api/scan", methods=["POST"])
def trigger_scan():
    """Trigger a manual scan via API."""
    return jsonify({"status": "Scan triggered via CLI. Use tray for full scan."}), 200


@app.route("/api/people")
@app.route("/people")
def list_people():
    from faces.face_clusterer import get_cluster_representatives

    repo = get_repo()
    clusters = get_cluster_representatives(repo)
    return jsonify({"clusters": clusters, "count": len(clusters)})


@app.route("/api/people/<cluster_id>")
@app.route("/people/<cluster_id>")
def get_person_cluster(cluster_id):
    repo = get_repo()
    items = repo.get_faces_by_cluster(cluster_id)
    if not items:
        return jsonify({"error": "Cluster not found"}), 404

    person_label = None
    face_count = 0
    for item in items:
        faces = item.get("faces", [])
        face_count += len(faces)
        if person_label is None:
            person_label = next(
                (face.get("person_label") for face in faces if face.get("person_label")),
                None,
            )

    return jsonify(
        {
            "cluster_id": cluster_id,
            "person_label": person_label,
            "face_count": face_count,
            "items": items,
        }
    )


@app.route("/api/people/<cluster_id>/label", methods=["POST"])
@app.route("/people/<cluster_id>/label", methods=["POST"])
def label_person_cluster(cluster_id):
    payload = request.get_json(silent=True) or {}
    label = str(payload.get("label", "")).strip()
    if not label:
        return jsonify({"error": "Label is required"}), 400

    repo = get_repo()
    updated = repo.update_person_label(cluster_id, label)
    if updated == 0:
        return jsonify({"error": "Cluster not found or unchanged"}), 404

    return jsonify({"cluster_id": cluster_id, "label": label, "updated": updated})


@app.route("/api/people/recluster", methods=["POST"])
@app.route("/people/recluster", methods=["POST"])
def recluster_people():
    from faces.face_clusterer import cluster_faces

    repo = get_repo()
    eps = request.args.get("eps", default=0.4, type=float)
    min_samples = request.args.get("min_samples", default=2, type=int)
    summary = cluster_faces(repo, eps=eps, min_samples=min_samples)
    return jsonify(summary)


@app.route("/image")
def serve_image():
    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        abort(404)
    return send_file(path)


@app.route("/face")
def serve_face_crop():
    from PIL import Image

    path = request.args.get("path", "")
    bbox = request.args.get("bbox", "").strip()

    if not path or not os.path.isfile(path) or not bbox:
        abort(404)

    try:
        x1, y1, x2, y2 = [int(float(value)) for value in bbox.split(",")]
    except ValueError:
        return jsonify({"error": "Invalid bbox"}), 400

    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size

        pad_x = max(12, int((x2 - x1) * 0.2))
        pad_y = max(12, int((y2 - y1) * 0.2))
        crop_box = (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )
        face = img.crop(crop_box)

    buffer = io.BytesIO()
    face.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    return send_file(buffer, mimetype="image/jpeg")


if __name__ == "__main__":
    print("Starting standalone Flask server...")
    init_engines()
    print("Ready at http://localhost:5000")
    app.run(debug=False, port=5000, host="127.0.0.1")
