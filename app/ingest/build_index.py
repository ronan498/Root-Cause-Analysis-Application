"""Builds or updates the index from CSV data."""
from pathlib import Path
import argparse
from typing import List, Dict

from app.rca.config import CSV_PATH
from app.rca.embedder import OpenAIEmbedder
from app.rca.index import RCAIndex
from app.rca.data_access import read_csv_rows

def main(rebuild: bool = False):
    embedder = OpenAIEmbedder()
    index = RCAIndex()
    have = index.load()
    rows = read_csv_rows(CSV_PATH)

    if rebuild or not have:
        texts = [r["fault_description"] for r in rows]
        embs = embedder.embed_texts(texts)
        index.build(embs, rows)
        print("Built fresh index from", len(rows), "rows")
    else:
        existing_texts = set(r["fault_description"] for r in index.meta)
        new_rows: List[Dict] = [r for r in rows if r["fault_description"] not in existing_texts]
        if new_rows:
            embs = embedder.embed_texts([r["fault_description"] for r in new_rows])
            index.add(embs, new_rows)
            print("Appended", len(new_rows), "new rows")
        else:
            print("No new rows to add")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    args = parser.parse_args()
    main(rebuild=args.rebuild)
