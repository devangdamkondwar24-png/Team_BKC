"""
Precompute Pipeline — Offline embedding generation + feature caching.

This runs ONCE, outside the 5-minute budget. Produces:
  precomputed/candidate_embeddings.npy  — (100K, 768) float32 matrix
  precomputed/candidate_ids.npy         — (100K,) string array
  precomputed/jd_embedding.npy          — (768,) float32 vector
  precomputed/company_earliest_starts.json — {company: earliest_start_date}

Usage:
  python precompute.py --candidates ./candidates.jsonl
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import EMBEDDING_MODEL, PRECOMPUTED_DIR, EMBEDDINGS_FILE, \
    CANDIDATE_IDS_FILE, JD_EMBEDDING_FILE, COMPANY_STARTS_FILE


# ============================================================================
# JD Text — the full job description distilled to embedding-friendly text
# ============================================================================
JD_TEXT = """
Senior AI Engineer, Founding Team at Redrob AI. Series A AI-native talent
intelligence platform. Location: Pune/Noida, India.

Core requirements: Production experience with embeddings-based retrieval
systems (sentence-transformers, OpenAI embeddings, BGE, E5) deployed to real
users. Production experience with vector databases or hybrid search
infrastructure (Pinecone, Weaviate, Qdrant, Milvus, Elasticsearch, FAISS).
Strong Python and code quality. Hands-on experience designing evaluation
frameworks for ranking systems (NDCG, MRR, MAP, A/B testing).

Role: Own the intelligence layer — ranking, retrieval, and matching systems
for candidate-JD matching at scale. Ship a v2 ranking system with embeddings,
hybrid retrieval, and LLM-based re-ranking. Set up evaluation infrastructure
with offline benchmarks and online A/B testing.

Ideal candidate: 6-8 years total experience, 4-5 years in applied ML/AI at
product companies. Shipped end-to-end ranking, search, or recommendation
system to real users at meaningful scale. Strong opinions on retrieval
(hybrid vs dense), evaluation (offline vs online), and LLM integration.
Located in or willing to relocate to Noida or Pune.

Nice to have: LLM fine-tuning (LoRA, QLoRA, PEFT), learning-to-rank models
(XGBoost), HR-tech/recruiting/marketplace products, distributed systems,
open-source contributions.

Do NOT want: pure research without production, recent LangChain-only
experience without pre-LLM ML background, title-chasers who job-hop every
1.5 years, career entirely at IT-services firms (TCS/Infosys/Wipro),
CV/speech/robotics specialists without NLP/IR, closed-source work with
no external validation.
"""


def load_candidates(path: str):
    """Load candidates from JSONL file, yielding (candidate_dict, line_idx)."""
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line), idx


def build_candidate_text(candidate: dict) -> str:
    """Build embedding text from profile summary + career descriptions.

    Concatenates the most semantically meaningful fields for JD matching.
    """
    profile = candidate.get("profile", {})
    parts = []

    # Headline and summary
    parts.append(profile.get("headline", ""))
    parts.append(profile.get("summary", ""))

    # Career descriptions (most important for semantic matching)
    for job in candidate.get("career_history", []):
        title = job.get("title", "")
        company = job.get("company", "")
        desc = job.get("description", "")
        parts.append(f"{title} at {company}: {desc}")

    # Key skills (names only)
    skill_names = [s.get("name", "") for s in candidate.get("skills", [])]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    return " ".join(parts)


def build_company_earliest_starts(candidates_path: str) -> dict:
    """Build lookup of earliest observed start_date per company across all candidates."""
    company_starts = {}
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            for job in c.get("career_history", []):
                company = job.get("company", "")
                start = job.get("start_date", "")
                if not company or not start:
                    continue
                if company not in company_starts or start < company_starts[company]:
                    company_starts[company] = start
    return company_starts


def main():
    parser = argparse.ArgumentParser(description="Precompute embeddings and features")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--output-dir", default=PRECOMPUTED_DIR, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    print(f"[1/5] Loading embedding model: {EMBEDDING_MODEL}")
    t0 = time.time()
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Step 2: Build candidate texts
    print(f"[2/5] Building candidate texts from {args.candidates}")
    t0 = time.time()
    candidate_ids = []
    candidate_texts = []
    for cand, idx in load_candidates(args.candidates):
        candidate_ids.append(cand["candidate_id"])
        candidate_texts.append(build_candidate_text(cand))
        if (idx + 1) % 10000 == 0:
            print(f"  Loaded {idx + 1} candidates...")
    print(f"  {len(candidate_ids)} candidates loaded in {time.time() - t0:.1f}s")

    # Step 3: Compute embeddings
    print(f"[3/5] Computing embeddings (batch_size={args.batch_size})...")
    t0 = time.time()
    embeddings = model.encode(
        candidate_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  Embeddings computed in {time.time() - t0:.1f}s, shape={embeddings.shape}")

    # Step 4: Compute JD embedding
    print("[4/5] Computing JD embedding...")
    jd_embedding = model.encode(
        [JD_TEXT],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    print(f"  JD embedding shape: {jd_embedding.shape}")

    # Step 5: Build company earliest starts
    print("[5/5] Building company earliest-start lookup...")
    t0 = time.time()
    company_starts = build_company_earliest_starts(args.candidates)
    print(f"  {len(company_starts)} companies indexed in {time.time() - t0:.1f}s")

    # Save everything
    print("Saving precomputed artifacts...")
    np.save(str(out_dir / EMBEDDINGS_FILE), embeddings.astype(np.float32))
    np.save(str(out_dir / CANDIDATE_IDS_FILE), np.array(candidate_ids))
    np.save(str(out_dir / JD_EMBEDDING_FILE), jd_embedding.astype(np.float32))

    with open(str(out_dir / COMPANY_STARTS_FILE), "w") as f:
        json.dump(company_starts, f)

    print(f"\nAll artifacts saved to {out_dir}/")
    print(f"  {EMBEDDINGS_FILE}: {embeddings.shape}")
    print(f"  {CANDIDATE_IDS_FILE}: {len(candidate_ids)} IDs")
    print(f"  {JD_EMBEDDING_FILE}: {jd_embedding.shape}")
    print(f"  {COMPANY_STARTS_FILE}: {len(company_starts)} companies")


if __name__ == "__main__":
    main()
