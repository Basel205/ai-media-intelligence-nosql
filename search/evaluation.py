"""
Evaluation Module — Baseline vs CSS Precision Benchmark
═══════════════════════════════════════════════════════
Uses Intel dataset folder labels as ground truth ONLY for evaluation.
Labels are never used during search — the system is fully unsupervised.

The evaluation answers: "Of the top-K results returned for a natural language
query, how many images actually belong to the correct category?"

Metric: Precision@K = relevant_results_in_top_K / K

Output: comparison table of Baseline (cosine) vs CSS.
This table goes directly into the project report.
"""

import os
from search.semantic_search import SemanticSearch
from search.hybrid_search import CompositeSemanticSearch


# Natural language query → ground truth folder label
EVAL_QUERIES = {
    "urban buildings and architecture":  "buildings",
    "dense forest with trees":           "forest",
    "glacier ice and snow":              "glacier",
    "mountain peak landscape":           "mountain",
    "sea ocean water waves":             "sea",
    "street road city traffic":          "street",
}

KNOWN_LABELS = {"buildings", "forest", "glacier", "mountain", "sea", "street"}


def get_label(file_path: str) -> str:
    """
    Extract ground truth label from the image's folder path.
    Works for any path that contains a category folder name anywhere in it.
    e.g. C:/.../ intel_dataset/seg_test/seg_test/forest/123.jpg → "forest"
    """
    # Normalise separators and split
    parts = file_path.replace("\\", "/").lower().split("/")
    for part in reversed(parts[:-1]):   # skip the filename itself
        if part in KNOWN_LABELS:
            return part
    return "unknown"


def precision_at_k(results: list, true_label: str, k: int) -> float:
    hits = sum(1 for r in results[:k] if get_label(r["file_path"]) == true_label)
    return round(hits / k, 3) if k > 0 else 0.0


def run_evaluation(
    baseline_engine=None,
    css_engine=None,
    k_values=(5, 10)
) -> list[dict]:
    """
    Run full evaluation. Accepts optional pre-loaded engines to avoid
    loading CLIP twice. Prints results table and returns raw data.
    """
    # Use passed engines (from app.py) or create new ones
    baseline = baseline_engine if baseline_engine else SemanticSearch()
    css      = css_engine      if css_engine      else CompositeSemanticSearch()

    max_k   = max(k_values)
    records = []

    print("\n" + "═" * 72)
    print(f"  {'Query':<38} {'K':>3}  {'Baseline':>9}  {'CSS':>9}  {'Δ':>7}")
    print("═" * 72)

    for query, label in EVAL_QUERIES.items():
        b_results = baseline.search(query, top_k=max_k)
        c_results = css.search(query,      top_k=max_k)

        # Debug: show what labels we're actually finding
        found_labels = set(get_label(r["file_path"]) for r in b_results)
        # print(f"  [debug] '{query}' → found labels: {found_labels}", flush=True)

        for k in k_values:
            p_b = precision_at_k(b_results, label, k)
            p_c = precision_at_k(c_results, label, k)
            d   = round(p_c - p_b, 3)
            d_s = f"+{d:.3f}" if d >= 0 else f"{d:.3f}"
            print(f"  {query[:38]:<38} {k:>3}  {p_b:>9.3f}  {p_c:>9.3f}  {d_s:>7}")
            records.append({"query": query, "label": label, "k": k,
                             "baseline": p_b, "css": p_c, "delta": d})

    print("─" * 72)
    for k in k_values:
        sub   = [r for r in records if r["k"] == k]
        avg_b = round(sum(r["baseline"] for r in sub) / len(sub), 3)
        avg_c = round(sum(r["css"]      for r in sub) / len(sub), 3)
        avg_d = round(avg_c - avg_b, 3)
        d_s   = f"+{avg_d:.3f}" if avg_d >= 0 else f"{avg_d:.3f}"
        print(f"  {'AVERAGE':<38} {k:>3}  {avg_b:>9.3f}  {avg_c:>9.3f}  {d_s:>7}")
    print("═" * 72 + "\n")

    return records


if __name__ == "__main__":
    run_evaluation()