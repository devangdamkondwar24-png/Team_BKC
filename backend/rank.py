"""
rank.py — Main ranking script

Produces the submission CSV from pre-computed embeddings + candidates.jsonl

Runtime target: <5 minutes on CPU, ≤16 GB RAM, no network
Usage: python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Pipeline:
  1. Load pre-computed embeddings + candidate metadata (streaming)
  2. Run Lane A (semantic), Lane B (career), Lane C (behavioral) in parallel
  3. Apply honeypot detection + hard disqualifiers
  4. Weighted composite score → top 500 candidates
  5. LightGBM lambdarank re-ranker on top 500 → top 100
  6. Generate per-candidate reasoning
  7. Write validated CSV
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import orjson
from tqdm import tqdm

from src.honeypot_detector import score_honeypot_confidence
from src.disqualifier import get_disqualifier_multiplier
from src.feature_extractor.semantic_scorer import SemanticScorer
from src.feature_extractor.career_scorer import score_candidate_career
from src.feature_extractor.behavioral_scorer import score_candidate_behavioral
from src.reranker import LambdaRankReranker, FEATURE_COLUMNS
from src.reasoning_generator import generate_reasoning


LANE_WEIGHTS = {"semantic": 0.45, "career": 0.35, "behavioral": 0.20}
TOP_N_RERANK = 500  # Re-rank top 500 with LightGBM


def load_candidates(path: str) -> List[Dict[str, Any]]:
    """Streaming JSONL load with orjson (handles both JSON and JSONL)."""
    with open(path, "rb") as f:
        sample = f.read(10)
        f.seek(0)
        if sample.strip().startswith(b"["):
            return orjson.loads(f.read())
        
        candidates = []
        for line in tqdm(f, desc="Loading candidates", unit="cand"):
            if line.strip():
                candidates.append(orjson.loads(line))
    return candidates


def build_feature_row(semantic_score: float,
                       career_out: Dict, behavioral_out: Dict) -> List[float]:
    """Build feature vector in FEATURE_COLUMNS order for LightGBM."""
    return [
        semantic_score,
        career_out.get("career_score", 0.0),
        behavioral_out.get("behavioral_score", 0.0),
        career_out.get("company_score", 0.0),
        career_out.get("title_score", 0.0),
        career_out.get("production_score", 0.0),
        career_out.get("ml_system_score", 0.0),
        career_out.get("yoe_score", 0.0),
        behavioral_out.get("recency_score", 0.0),
        behavioral_out.get("response_score", 0.0),
        behavioral_out.get("notice_score", 0.0),
        behavioral_out.get("location_score", 0.0),
        behavioral_out.get("interview_score", 0.0),
        behavioral_out.get("github_score", 0.0),
        behavioral_out.get("completeness_score", 0.0),
        behavioral_out.get("saved_score", 0.0),
        behavioral_out.get("open_to_work", 0.0),
    ]


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--embeddings-dir", default="data", help="Dir with pre-computed embeddings")
    parser.add_argument("--top-n-rerank", type=int, default=TOP_N_RERANK)
    args = parser.parse_args()

    start_time = time.time()
    print(f"[rank.py] Starting ranking pipeline...")
    print(f"[rank.py] Candidates: {args.candidates}")
    print(f"[rank.py] Output: {args.out}")

    # ── Stage 1: Load candidates ───────────────────────────────────────────────
    print("\n[Stage 1] Loading candidates...")
    t1 = time.time()
    candidates = load_candidates(args.candidates)
    print(f"[Stage 1] Loaded {len(candidates):,} candidates in {time.time()-t1:.1f}s")

    # ── Stage 2: Load pre-computed embeddings + compute semantic scores ────────
    print("\n[Stage 2] Computing semantic scores (BGE + BM25 hybrid)...")
    t2 = time.time()
    scorer = SemanticScorer(
        embeddings_path=f"{args.embeddings_dir}/embeddings.npy",
        candidate_ids_path=f"{args.embeddings_dir}/candidate_ids.json"
    )
    scorer.load_precomputed()
    semantic_scores = scorer.score_all(candidates)
    print(f"[Stage 2] Semantic scoring done in {time.time()-t2:.1f}s")

    # ── Stage 3: Per-candidate feature extraction ─────────────────────────────
    print("\n[Stage 3] Extracting career + behavioral features...")
    t3 = time.time()

    all_scores = {}
    career_outputs = {}
    behavioral_outputs = {}

    for candidate in tqdm(candidates, desc="Feature extraction", unit="cand"):
        cid = candidate["candidate_id"]

        # Honeypot + disqualifier checks
        honeypot_mult = score_honeypot_confidence(candidate)
        disq_mult = get_disqualifier_multiplier(candidate)

        if honeypot_mult == 0.0 or disq_mult == 0.0:
            all_scores[cid] = 0.0
            career_outputs[cid] = {}
            behavioral_outputs[cid] = {}
            continue

        # Score all lanes
        sem = semantic_scores.get(cid, 0.0)
        career_out = score_candidate_career(candidate)
        behavioral_out = score_candidate_behavioral(candidate)

        # Weighted composite
        composite = (
            LANE_WEIGHTS["semantic"] * sem +
            LANE_WEIGHTS["career"] * career_out["career_score"] +
            LANE_WEIGHTS["behavioral"] * behavioral_out["behavioral_score"]
        )

        all_scores[cid] = composite
        career_outputs[cid] = career_out
        behavioral_outputs[cid] = behavioral_out

    print(f"[Stage 3] Feature extraction done in {time.time()-t3:.1f}s")

    # ── Stage 4: Filter top-N for re-ranking ──────────────────────────────────
    print(f"\n[Stage 4] Selecting top-{args.top_n_rerank} for LightGBM re-ranking...")
    sorted_ids = sorted(all_scores.keys(), key=lambda x: (-all_scores[x], x))
    top_n_ids = sorted_ids[:args.top_n_rerank]

    # Build feature matrix
    top_n_candidates_map = {c["candidate_id"]: c for c in candidates if c["candidate_id"] in set(top_n_ids)}
    features = []
    composite_scores = []

    for cid in top_n_ids:
        sem = semantic_scores.get(cid, 0.0)
        career_out = career_outputs.get(cid, {})
        behavioral_out = behavioral_outputs.get(cid, {})
        features.append(build_feature_row(sem, career_out, behavioral_out))
        composite_scores.append(all_scores[cid])

    features_array = np.array(features, dtype=np.float32)
    composite_array = np.array(composite_scores, dtype=np.float32)

    # ── Stage 5: LightGBM LambdaRank re-ranking ───────────────────────────────
    print(f"\n[Stage 5] LightGBM LambdaRank re-ranking {len(top_n_ids)} candidates...")
    t5 = time.time()
    reranker = LambdaRankReranker()
    reranker.fit(features_array, composite_array, len(top_n_ids))
    rerank_scores = reranker.predict(features_array)
    print(f"[Stage 5] Re-ranking done in {time.time()-t5:.1f}s")

    # Sort by re-ranker scores: descending by score, ascending by candidate_id for tie-breaks
    reranked = sorted(zip(top_n_ids, rerank_scores), key=lambda x: (-x[1], x[0]))
    top_100 = reranked[:100]

    # ── Stage 6: Generate reasoning + write CSV ───────────────────────────────
    print(f"\n[Stage 6] Generating reasoning and writing CSV...")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, (cid, score) in enumerate(top_100, start=1):
            candidate = top_n_candidates_map.get(cid, {})
            career_out = career_outputs.get(cid, {})
            behavioral_out = behavioral_outputs.get(cid, {})
            reasoning = generate_reasoning(candidate, behavioral_out, career_out, rank, score)
            writer.writerow([cid, rank, f"{float(score):.6f}", reasoning])

    elapsed = time.time() - start_time
    print(f"\n[rank.py] Complete in {elapsed:.1f}s")
    print(f"[rank.py] Output: {args.out}")
    print(f"[rank.py] Top candidate: {top_100[0][0]} (score: {top_100[0][1]:.4f})")

    # Warn if over time budget
    if elapsed > 280:
        print(f"[rank.py] ⚠️  WARNING: {elapsed:.0f}s — approaching 5-minute limit")


if __name__ == "__main__":
    main()
