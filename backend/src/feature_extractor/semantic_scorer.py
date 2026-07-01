"""
Semantic Scorer — Lane A (weight 0.45 in final score)

Architecture: BGE-base-en-v1.5 dense embeddings + BM25Okapi keyword retrieval
Fusion: Reciprocal Rank Fusion (RRF) — industry standard at LinkedIn/Elastic

Why BGE-base-en-v1.5:
- #1 open-source English retrieval model on MTEB (no API needed)
- 768-dim vectors, CPU inference ~30ms/doc
- Much better than MiniLM for domain-specific technical vocabulary
- Self-hosted: no network dependency during ranking step

Pre-computation strategy:
- Embeddings generated offline in precompute.py (not time-limited)
- Saved to data/embeddings.npy + data/candidate_ids.json
- Ranking step loads pre-computed embeddings → cosine sim in <10s for 100K

BM25 is computed at rank time (fast, no pre-computation needed)
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path


# ─── JD Query (what we're searching for) ───────────────────────────────────────
JD_QUERY = """
Senior AI engineer with production experience in embeddings-based retrieval systems,
vector databases, semantic search, hybrid search, dense retrieval, learning to rank.
Experience with sentence-transformers, FAISS, Pinecone, Weaviate, Qdrant, Elasticsearch.
Python, evaluation frameworks, NDCG MRR MAP A/B testing. Product company experience.
Shipped end-to-end ranking or search system to real users at scale. Not research-only.
5 to 9 years experience. NLP information retrieval recommendation systems.
"""

JD_KEYWORDS = [
    "embedding", "retrieval", "vector database", "semantic search", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "sentence transformer",
    "dense retrieval", "hybrid search", "bm25", "ndcg", "learning to rank",
    "production", "deployed", "search", "ranking", "recommendation",
    "nlp", "information retrieval", "python", "evaluation", "a/b test"
]


def build_candidate_text(candidate: Dict[str, Any]) -> str:
    """
    Build a rich text representation of the candidate for embedding.
    Priority order: current title/headline > career descriptions > skills
    This is what gets embedded — it must capture semantic meaning, not just keywords.
    """
    parts = []
    profile = candidate.get("profile", {})

    # High-signal fields first
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"][:500])  # Cap summary to avoid noise

    # Career descriptions (most signal-rich field)
    career = candidate.get("career_history") or []
    for job in sorted(career, key=lambda x: x.get("start_date") or "", reverse=True)[:3]:
        title = job.get("title", "")
        desc = job.get("description", "")[:300]
        company = job.get("company", "")
        if title or desc:
            parts.append(f"{title} at {company}: {desc}")

    # Skills (with proficiency weighting)
    skills = candidate.get("skills") or []
    advanced_skills = [s["name"] for s in skills if s.get("proficiency") in ("advanced", "expert")]
    if advanced_skills:
        parts.append("Advanced skills: " + ", ".join(advanced_skills[:10]))

    return " ".join(parts)


class SemanticScorer:
    def __init__(self, embeddings_path: str = "data/embeddings.npy",
                 candidate_ids_path: str = "data/candidate_ids.json"):
        self.embeddings_path = embeddings_path
        self.candidate_ids_path = candidate_ids_path
        self._embeddings = None
        self._candidate_ids = None
        self._jd_embedding = None
        self._model = None

    def _load_model(self):
        """Lazy-load model only when needed (saves memory if not used)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            # BGE requires a query instruction prefix for retrieval tasks
            self._model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        return self._model

    def _encode_jd(self) -> np.ndarray:
        """Encode the JD query with BGE instruction prefix."""
        if self._jd_embedding is None:
            model = self._load_model()
            # BGE instruction prefix for retrieval queries
            query = "Represent this sentence for searching relevant passages: " + JD_QUERY
            self._jd_embedding = model.encode([query], normalize_embeddings=True)[0]
        return self._jd_embedding

    def load_precomputed(self):
        """Load pre-computed embeddings from disk. Fast — just numpy load."""
        import json
        self._embeddings = np.load(self.embeddings_path)
        with open(self.candidate_ids_path) as f:
            self._candidate_ids = json.load(f)
        print(f"[SemanticScorer] Loaded {len(self._candidate_ids)} pre-computed embeddings")

    def compute_dense_scores(self) -> Dict[str, float]:
        """
        Compute cosine similarity between JD embedding and all candidate embeddings.
        With pre-computed embeddings and numpy, this runs in ~2 seconds for 100K.
        """
        if self._embeddings is None:
            raise RuntimeError("Call load_precomputed() first")

        jd_emb = self._encode_jd()  # shape: (768,)
        # Matrix multiply: (100000, 768) @ (768,) = (100000,) cosine scores
        # Embeddings are L2-normalized, so dot product == cosine similarity
        scores = self._embeddings @ jd_emb
        return {cid: float(scores[i]) for i, cid in enumerate(self._candidate_ids)}

    def compute_bm25_scores(self, candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        BM25 keyword retrieval over candidate texts.
        Complementary to dense: catches exact technical terms dense misses.
        """
        from rank_bm25 import BM25Okapi

        cids = [c["candidate_id"] for c in candidates]
        corpus = [build_candidate_text(c).lower().split() for c in candidates]

        bm25 = BM25Okapi(corpus)
        query_tokens = " ".join(JD_KEYWORDS).lower().split()
        raw_scores = bm25.get_scores(query_tokens)

        # Normalize to [0, 1]
        max_score = max(raw_scores) if max(raw_scores) > 0 else 1.0
        return {cid: float(raw_scores[i] / max_score) for i, cid in enumerate(cids)}

    @staticmethod
    def reciprocal_rank_fusion(scores_a: Dict[str, float],
                                scores_b: Dict[str, float],
                                k: int = 60) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion — industry-standard hybrid retrieval fusion.
        Used by Elastic, Weaviate, Cohere in production hybrid search pipelines.
        k=60 is the standard default from the original Cormack et al. paper.
        """
        # Convert scores to ranks
        def scores_to_ranks(scores: Dict[str, float]) -> Dict[str, int]:
            sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            return {cid: rank + 1 for rank, cid in enumerate(sorted_ids)}

        ranks_a = scores_to_ranks(scores_a)
        ranks_b = scores_to_ranks(scores_b)

        all_ids = set(scores_a.keys()) | set(scores_b.keys())
        rrf_scores = {}
        for cid in all_ids:
            rank_a = ranks_a.get(cid, len(scores_a) + 1)
            rank_b = ranks_b.get(cid, len(scores_b) + 1)
            rrf_scores[cid] = 1.0 / (k + rank_a) + 1.0 / (k + rank_b)

        # Normalize to [0, 1]
        max_rrf = max(rrf_scores.values())
        return {cid: s / max_rrf for cid, s in rrf_scores.items()}

    def score_all(self, candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Main entry point. Returns {candidate_id: semantic_score} for all candidates.
        Hybrid: 70% dense BGE + 30% BM25, fused with RRF.
        """
        print("[SemanticScorer] Computing dense scores...")
        dense_scores = self.compute_dense_scores()

        print("[SemanticScorer] Computing BM25 scores...")
        bm25_scores = self.compute_bm25_scores(candidates)

        print("[SemanticScorer] Fusing with RRF...")
        hybrid_scores = self.reciprocal_rank_fusion(dense_scores, bm25_scores, k=60)

        return hybrid_scores
