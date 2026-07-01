"""
precompute.py — Offline embedding generation (run once, before competition)

This generates BGE embeddings for all 100K candidates and saves to disk.
The ranking step (rank.py) loads these pre-computed embeddings — fast numpy.

Runtime: ~25-40 min on CPU for 100K candidates
Memory: ~600MB for embeddings.npy (100K × 768 float32)

Usage: python precompute.py --candidates candidates.jsonl
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import orjson
from sentence_transformers import SentenceTransformer

from src.feature_extractor.semantic_scorer import build_candidate_text


def precompute_embeddings(candidates_path: str, output_dir: str = "data",
                           batch_size: int = 256):
    Path(output_dir).mkdir(exist_ok=True)

    # Load model
    print("[Precompute] Loading BAAI/bge-base-en-v1.5...")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # Stream candidates
    print(f"[Precompute] Loading candidates from {candidates_path}...")
    with open(candidates_path, "rb") as f:
        sample = f.read(10)
        f.seek(0)
        if sample.strip().startswith(b"["):
            candidates = orjson.loads(f.read())
        else:
            candidates = []
            for line in f:
                if line.strip():
                    candidates.append(orjson.loads(line))

    print(f"[Precompute] Loaded {len(candidates):,} candidates")

    candidate_ids = [c["candidate_id"] for c in candidates]
    texts = [build_candidate_text(c) for c in candidates]

    # Encode in batches with progress bar
    print(f"[Precompute] Encoding {len(texts):,} texts in batches of {batch_size}...")
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(
            batch,
            normalize_embeddings=True,  # L2 normalize for cosine sim via dot product
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.append(embeddings)

    embeddings_matrix = np.vstack(all_embeddings).astype(np.float32)
    print(f"[Precompute] Embeddings shape: {embeddings_matrix.shape}")

    # Save
    np.save(f"{output_dir}/embeddings.npy", embeddings_matrix)
    with open(f"{output_dir}/candidate_ids.json", "w") as f:
        json.dump(candidate_ids, f)

    print(f"[Precompute] Saved to {output_dir}/embeddings.npy and {output_dir}/candidate_ids.json")
    print(f"[Precompute] File size: {embeddings_matrix.nbytes / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    precompute_embeddings(args.candidates, args.output_dir, args.batch_size)
