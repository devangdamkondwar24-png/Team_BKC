"""
Composite Scorer — Combines all pillars with multiplicative disqualifier.

Formula:
  raw = W_SEMANTIC * semantic + W_CAREER * career + W_STRUCTURED * structured + W_BEHAVIORAL * behavioral
  final = raw * disqualifier_multiplier * (0.0 if honeypot else 1.0)
"""
from src.config import W_SEMANTIC, W_CAREER_DEPTH, W_STRUCTURED_FIT, W_BEHAVIORAL
from src.career_analyzer import career_depth_score
from src.feature_extractor import structured_fit_score
from src.behavioral_scorer import behavioral_score
from src.disqualifier import disqualifier_multiplier
from src.honeypot_detector import is_honeypot


def composite_score(
    candidate: dict,
    semantic_sim: float,
    company_earliest_starts: dict | None = None,
) -> tuple[float, dict]:
    """Compute final composite score for a candidate.

    Returns:
        (score, breakdown) where breakdown has all sub-scores for debugging/reasoning.
    """
    career = career_depth_score(candidate)
    structured = structured_fit_score(candidate)
    behavioral = behavioral_score(candidate)

    raw_score = (
        W_SEMANTIC * semantic_sim
        + W_CAREER_DEPTH * career
        + W_STRUCTURED_FIT * structured
        + W_BEHAVIORAL * behavioral
    )

    disq_mult = disqualifier_multiplier(candidate)
    honeypot = is_honeypot(candidate, company_earliest_starts)

    if honeypot:
        final = 0.0
    else:
        final = raw_score * disq_mult

    breakdown = {
        "semantic": semantic_sim,
        "career_depth": career,
        "structured_fit": structured,
        "behavioral": behavioral,
        "raw_score": raw_score,
        "disqualifier_multiplier": disq_mult,
        "is_honeypot": honeypot,
        "final_score": final,
    }

    return final, breakdown
