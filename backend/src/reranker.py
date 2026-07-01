"""
LightGBM LambdaRank Re-Ranker — Stage 5

This is the same architecture LinkedIn and Airbnb use in production:
  - Two-tower / embedding retrieval generates top-N candidates
  - LightGBM lambdarank re-ranks them with all available features

Since we have no ground-truth labels, we generate pseudo-labels from
the three-lane composite score. LightGBM then re-orders based on feature
interactions that simple weighted sum misses.

Key insight: LambdaRank directly optimizes NDCG — the primary metric.
A simple weighted average optimizes for correlation, not ranking quality.
"""

import numpy as np
import lightgbm as lgb
from typing import Dict, List, Any


FEATURE_COLUMNS = [
    "semantic_score",         # Lane A: BGE + BM25 hybrid
    "career_score",           # Lane B: career narrative
    "behavioral_score",       # Lane C: availability signals
    "company_score",          # product vs consulting
    "title_score",            # title relevance tier
    "production_score",       # production deployment keywords
    "ml_system_score",        # ML/IR system keywords
    "yoe_score",              # YOE band fit
    "recency_score",          # days since last active
    "response_score",         # recruiter response rate
    "notice_score",           # notice period
    "location_score",         # location fit
    "interview_score",        # interview completion
    "github_score",           # github activity
    "completeness_score",     # profile completeness
    "saved_score",            # saved by recruiters 30d
    "open_to_work",           # open to work flag
]


def build_pseudo_labels(composite_scores: np.ndarray) -> np.ndarray:
    """
    Generate pseudo-labels from composite scores for LambdaRank training.
    LambdaRank needs relative relevance labels (higher = more relevant).

    Tier mapping:
      composite >= 0.75 → tier 4 (excellent fit)
      composite >= 0.60 → tier 3 (strong fit)
      composite >= 0.45 → tier 2 (moderate fit)
      composite >= 0.30 → tier 1 (weak fit)
      composite <  0.30 → tier 0 (not fit)
    """
    labels = np.zeros(len(composite_scores), dtype=np.int32)
    labels[composite_scores >= 0.30] = 1
    labels[composite_scores >= 0.45] = 2
    labels[composite_scores >= 0.60] = 3
    labels[composite_scores >= 0.75] = 4
    return labels


class LambdaRankReranker:
    def __init__(self):
        self.model = None

    def fit(self, features: np.ndarray, composite_scores: np.ndarray, n_candidates: int):
        """
        Train LightGBM LambdaRank on top-N candidates.
        Features: (N, len(FEATURE_COLUMNS)) matrix
        composite_scores: (N,) array for pseudo-label generation
        """
        labels = build_pseudo_labels(composite_scores)

        # LightGBM lambdarank requires group information
        # Since all candidates are from one "query" (the JD), group size = N
        group = [n_candidates]

        train_data = lgb.Dataset(
            features,
            label=labels,
            group=group,
            feature_name=FEATURE_COLUMNS,
            free_raw_data=False
        )

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10, 50],   # Matches competition metrics
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 5,
            "lambda_l2": 0.1,
            "verbose": -1,
            "n_jobs": -1,
        }

        print("[Reranker] Training LightGBM LambdaRank on top-N candidates...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(50)]
        )
        print("[Reranker] Training complete.")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Returns re-ranked scores."""
        if self.model is None:
            raise RuntimeError("Call fit() first")
        return self.model.predict(features)
