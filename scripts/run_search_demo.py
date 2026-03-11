"""
Demo script — runs CSS vs baseline comparison for a query.
Usage:
    python scripts/run_search_demo.py "sea waves"
    python scripts/run_search_demo.py "forest trees" 10
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.semantic_search import SemanticSearch
from search.hybrid_search import CompositeSemanticSearch

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "nature landscape"
    k     = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print(f'\nQuery: "{query}"  |  Top {k} results\n')

    print("── Baseline (cosine only) ──────────────────────────────")
    for i, r in enumerate(SemanticSearch().search(query, top_k=k), 1):
        print(f"  {i}. [{r['cosine_similarity']:.4f}]  {r['file_name']}")

    print("\n── CSS (composite score) ───────────────────────────────")
    for i, r in enumerate(CompositeSemanticSearch().search(query, top_k=k), 1):
        b = r["_breakdown"]
        print(f"  {i}. [{r['css_score']:.4f}]  {r['file_name']}"
              f"  (cos={b['cosine']:.3f}  sharp={b['sharpness_norm']:.3f}  bright={b['brightness_norm']:.3f})")

    print()