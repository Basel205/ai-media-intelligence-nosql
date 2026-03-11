"""
Media Intelligence CLI
======================
python main.py setup          — initialise database schema
python main.py ingest         — scan and ingest images
python main.py embed          — generate CLIP embeddings
python main.py search "query" — CSS search (default)
python main.py search "query" --baseline  — cosine baseline
python main.py search "query" --k 20      — return 20 results
python main.py compare "query"            — side-by-side CSS vs baseline
python main.py duplicates     — find near-duplicate images
python main.py evaluate       — run Precision@K benchmark
python main.py stats          — database statistics
"""

import sys
import argparse
from utils.config import DATASET_PATH


def cmd_setup():
    from database.schema_setup import SchemaSetup
    SchemaSetup().setup()

def cmd_ingest():
    from ingestion.pipeline import MediaIngestionPipeline
    MediaIngestionPipeline(DATASET_PATH).run()

def cmd_embed():
    from embeddings.generate_embeddings import EmbeddingGenerator
    EmbeddingGenerator().run()

def cmd_search(query, baseline=False, top_k=10):
    if baseline:
        from search.semantic_search import SemanticSearch
        results = SemanticSearch().search(query, top_k=top_k)
        method  = "Baseline (cosine)"
        key     = "cosine_similarity"
    else:
        from search.hybrid_search import CompositeSemanticSearch
        results = CompositeSemanticSearch().search(query, top_k=top_k)
        method  = "CSS (composite)"
        key     = "css_score"

    print(f'\n{method} — "{query}"')
    print("─" * 55)
    for i, r in enumerate(results, 1):
        print(f"  {i:>2}. [{r[key]:.4f}]  {r['file_name']}")
    print()

def cmd_compare(query, top_k=6):
    from search.semantic_search import SemanticSearch
    from search.hybrid_search import CompositeSemanticSearch

    b = SemanticSearch().search(query, top_k=top_k)
    c = CompositeSemanticSearch().search(query, top_k=top_k)

    print(f'\nComparing: "{query}"')
    print(f"{'Rank':<5} {'Baseline (cosine)':<40} {'CSS':<40}")
    print("─" * 85)
    for i in range(top_k):
        br = b[i]["file_name"] + f" [{b[i]['cosine_similarity']:.3f}]" if i < len(b) else "—"
        cr = c[i]["file_name"] + f" [{c[i]['css_score']:.3f}]"         if i < len(c) else "—"
        print(f"  {i+1:<3} {br:<40} {cr:<40}")
    print()

def cmd_duplicates():
    from quality.duplicate_detector import DuplicateDetector
    DuplicateDetector().report()

def cmd_evaluate():
    from search.evaluation import run_evaluation
    run_evaluation()

def cmd_stats():
    from database.media_repository import MediaRepository
    repo = MediaRepository()
    total    = repo.count()
    embedded = repo.count_embedded()
    print(f"\nDatabase: media_intelligence")
    print(f"  Total documents : {total}")
    print(f"  With embeddings : {embedded}")
    print(f"  Pending embed   : {total - embedded}\n")


def main():
    parser = argparse.ArgumentParser(description="Media Intelligence CLI")
    parser.add_argument("command", choices=["setup","ingest","embed","search",
                                            "compare","duplicates","evaluate","stats"])
    parser.add_argument("query", nargs="?")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    if   args.command == "setup":      cmd_setup()
    elif args.command == "ingest":     cmd_ingest()
    elif args.command == "embed":      cmd_embed()
    elif args.command == "search":
        if not args.query: print("Error: provide a query."); sys.exit(1)
        cmd_search(args.query, baseline=args.baseline, top_k=args.k)
    elif args.command == "compare":
        if not args.query: print("Error: provide a query."); sys.exit(1)
        cmd_compare(args.query, top_k=args.k)
    elif args.command == "duplicates": cmd_duplicates()
    elif args.command == "evaluate":   cmd_evaluate()
    elif args.command == "stats":      cmd_stats()


if __name__ == "__main__":
    main()
