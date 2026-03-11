"""
Quick script to run the full ingestion pipeline.
Usage:
    python scripts/ingest_dataset.py
    python scripts/ingest_dataset.py "C:\Users\kbase\Pictures"
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.pipeline import MediaIngestionPipeline
from utils.config import DATASET_PATH

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else DATASET_PATH
    print(f"Ingesting from: {folder}")
    MediaIngestionPipeline(folder).run()