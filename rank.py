"""
rank.py — Main ranking entrypoint.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline (all within 5-minute CPU budget):
  1. Load precomputed embeddings + company starts          (~2-5 sec)
  2. Load all candidates from JSONL                        (~10-15 sec)
  3. Fast prefilter: drop non-tech + obvious honeypots     (~5 sec)
  4. Compute semantic similarities (vectorized)            (~1 sec)
  5. Score ALL post-filter candidates (career+struct+behav) (~30-60 sec)
  6. Apply disqualifier multiplier + honeypot gate          (~5 sec)
  7. Sort, take top 100, generate reasoning                (~5 sec)
  8. Write submission.csv                                  (~1 sec)
"""
import argparse
import csv
import json
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PRECOMPUTED_DIR, EMBEDDINGS_FILE, CANDIDATE_IDS_FILE,
    JD_EMBEDDING_FILE, COMPANY_STARTS_FILE, NON_TECH_TITLES,
)
from src.semantic_matcher import compute_semantic_similarities, normalize_similarities
from src.composite_scorer import composite_score
from src.honeypot_detector import is_honeypot
from src.reasoning_generator import generate_reasoning


def load_candidates_dict(path: str) -> dict:
    """Load all candidates into a dict keyed by candidate_id."""
    candidates = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            candidates[c["candidate_id"]] = c
    return candidates


def fast_prefilter(candidate: dict) -> bool:
    """Quick check: should this candidate proceed to full scoring?

    Returns True if candidate should be KEPT (not filtered out).
    We only drop candidates whose title AND career are entirely non-tech,
    keeping the filter conservative to avoid losing plain-language fits.
    """
    title = candidate.get("profile", {}).get("current_title", "")

    # If they have a tech title, always keep
    if title not in NON_TECH_TITLES:
        return True

    # Non-tech title: check if any career role is tech-adjacent
    tech_keywords_in_desc = {
        "machine learning", "deep learning", "neural", "embedding",
        "model", "training", "inference", "nlp", "search", "ranking",
        "recommendation", "retrieval", "vector", "algorithm",
        "python", "deployed", "production ml", "data pipeline",
        "tensorflow", "pytorch", "scikit", "feature engineering",
    }
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").lower()
        if any(kw in desc for kw in tech_keywords_in_desc):
            return True

    # Non-tech title + non-tech career → filter out
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates against JD and produce submission.csv"
    )
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--precomputed-dir", default=PRECOMPUTED_DIR,
                        help="Directory with precomputed artifacts")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Number of top candidates to output")
    args = parser.parse_args()

    start_time = time.time()
    pre_dir = Path(args.precomputed_dir)

    # ── Step 1: Load precomputed artifacts ──────────────────────────────
    print("[1/8] Loading precomputed artifacts...")
    t0 = time.time()

    embeddings = np.load(str(pre_dir / EMBEDDINGS_FILE))
    candidate_ids_arr = np.load(str(pre_dir / CANDIDATE_IDS_FILE), allow_pickle=True)
    jd_embedding = np.load(str(pre_dir / JD_EMBEDDING_FILE))

    with open(str(pre_dir / COMPANY_STARTS_FILE), "r") as f:
        company_earliest_starts = json.load(f)

    # Build id→index map
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids_arr)}
    print(f"  Loaded {len(candidate_ids_arr)} embeddings in {time.time()-t0:.1f}s")

    # ── Step 2: Load all candidates ─────────────────────────────────────
    print("[2/8] Loading candidates from JSONL...")
    t0 = time.time()
    all_candidates = load_candidates_dict(args.candidates)
    print(f"  Loaded {len(all_candidates)} candidates in {time.time()-t0:.1f}s")

    # ── Step 3: Fast prefilter ──────────────────────────────────────────
    print("[3/8] Applying fast prefilter...")
    t0 = time.time()
    filtered_ids = []
    dropped = 0
    for cid, cand in all_candidates.items():
        if fast_prefilter(cand):
            filtered_ids.append(cid)
        else:
            dropped += 1
    print(f"  Kept {len(filtered_ids)}, dropped {dropped} in {time.time()-t0:.1f}s")

    # ── Step 4: Compute semantic similarities ───────────────────────────
    print("[4/8] Computing semantic similarities (vectorized)...")
    t0 = time.time()
    # Compute for ALL candidates (full matrix), then index into results
    all_similarities = compute_semantic_similarities(jd_embedding, embeddings)
    all_similarities = normalize_similarities(all_similarities)
    print(f"  Similarities computed in {time.time()-t0:.1f}s")

    # ── Step 5: Score ALL post-filter candidates ────────────────────────
    print(f"[5/8] Scoring {len(filtered_ids)} candidates...")
    t0 = time.time()
    scored = []
    for i, cid in enumerate(filtered_ids):
        cand = all_candidates[cid]
        idx = id_to_idx.get(cid)
        if idx is None:
            continue
        sem_sim = float(all_similarities[idx])
        score, breakdown = composite_score(
            cand, sem_sim, company_earliest_starts
        )
        scored.append((cid, score, breakdown, cand))

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {i+1}/{len(filtered_ids)} scored ({rate:.0f} cand/s)")

    print(f"  Scored {len(scored)} candidates in {time.time()-t0:.1f}s")

    # ── Step 6: Honeypot gate (already applied in composite_score) ──────
    print("[6/8] Filtering honeypots...")
    t0 = time.time()
    non_honeypot = [(cid, sc, bd, ca) for cid, sc, bd, ca in scored if not bd["is_honeypot"]]
    honeypot_count = len(scored) - len(non_honeypot)
    print(f"  Removed {honeypot_count} honeypots in {time.time()-t0:.1f}s")

    # ── Step 7: Sort and take top-K ─────────────────────────────────────
    print(f"[7/8] Sorting and selecting top {args.top_k}...")
    t0 = time.time()
    # Sort by score descending (rounded to 4 decimal places), then candidate_id ascending for ties
    non_honeypot.sort(key=lambda x: (-round(x[1], 4), x[0]))
    top_k = non_honeypot[:args.top_k]

    # Generate reasoning for top-K
    results = []
    for rank_idx, (cid, score, breakdown, cand) in enumerate(top_k):
        reasoning = generate_reasoning(cand, breakdown)
        results.append({
            "candidate_id": cid,
            "rank": rank_idx + 1,
            "score": round(score, 4),
            "reasoning": reasoning,
        })
    print(f"  Top {len(results)} selected in {time.time()-t0:.1f}s")

    # ── Step 8: Write CSV ───────────────────────────────────────────────
    print(f"[8/8] Writing {args.out}...")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(results)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Output: {args.out} ({len(results)} candidates)")
    print(f"Score range: {results[0]['score']:.4f} (rank 1) -> {results[-1]['score']:.4f} (rank {len(results)})")
    print(f"Honeypots filtered: {honeypot_count}")
    print(f"{'='*60}")

    if elapsed > 300:
        print("WARNING: Exceeded 5-minute budget!")
    else:
        print(f"OK: Within 5-minute budget ({300-elapsed:.0f}s remaining)")


if __name__ == "__main__":
    main()
