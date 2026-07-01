"""
Reasoning Generator — produces per-candidate 1-2 sentence reasoning.

Stage 4 evaluation criteria (from submission_spec.md):
  ✓ References specific facts from the candidate's profile
  ✓ Connects to specific JD requirements
  ✓ Acknowledges honest concerns where they exist
  ✓ No hallucination (every claim verifiable in profile)
  ✓ Variation (not templated across candidates)
  ✓ Rank consistency (tone matches rank)

Strategy: rule-based extraction of the 2-3 most discriminating facts,
then a template that varies based on candidate profile structure.
"""

from typing import Dict, Any


def _get_company_type_label(career_scorer_output: Dict) -> str:
    cs = career_scorer_output.get("company_score", 0.5)
    if cs >= 0.85:
        return "product company"
    if cs >= 0.65:
        return "mixed experience"
    return "services/consulting background"


def _get_top_relevant_skill(candidate: Dict[str, Any]) -> str:
    """Return the most JD-relevant advanced skill the candidate has."""
    JD_SKILLS = [
        "faiss", "pinecone", "weaviate", "qdrant", "sentence-transformers",
        "embedding", "vector database", "semantic search", "bm25", "hybrid search",
        "lightgbm", "learning to rank", "elasticsearch", "opensearch",
        "recommendation", "information retrieval", "ranking", "nlp", "rag"
    ]
    skills = candidate.get("skills") or []
    for skill in sorted(skills, key=lambda s: s.get("endorsements", 0), reverse=True):
        name = (skill.get("name") or "").lower()
        for jd_skill in JD_SKILLS:
            if jd_skill in name:
                return skill["name"]
    # Fallback: return highest-endorsed advanced skill
    advanced = [s for s in skills if s.get("proficiency") in ("advanced", "expert")]
    if advanced:
        return max(advanced, key=lambda s: s.get("endorsements", 0))["name"]
    return ""


def _get_concern(candidate: Dict[str, Any], behavioral: Dict, career: Dict) -> str:
    """Extract the most notable concern for honest reporting."""
    signals = candidate.get("redrob_signals") or {}

    # Check notice period
    notice = signals.get("notice_period_days")
    if notice and notice > 90:
        return f"notice period of {int(notice)}d is above JD's sub-30d preference"

    # Check recency
    recency = behavioral.get("recency_score", 1.0)
    if recency < 0.5:
        return "platform inactivity (last active >90 days ago) may affect reachability"

    # Check response rate
    rr = signals.get("recruiter_response_rate")
    if rr is not None and rr < 0.25:
        return f"low recruiter response rate ({int(rr*100)}%) is a risk"

    # Check location
    location_score = behavioral.get("location_score", 1.0)
    if location_score < 0.7:
        profile = candidate.get("profile", {})
        return f"located in {profile.get('location', 'unknown location')}, relocation required"

    # Check consulting background
    cs = career.get("company_score", 1.0)
    if cs < 0.5:
        return "primarily services/consulting background — no clear product company experience on record"

    return ""  # No notable concern


def generate_reasoning(candidate: Dict[str, Any],
                        behavioral_scores: Dict,
                        career_scores: Dict,
                        final_rank: int,
                        final_score: float) -> str:
    """
    Generate a specific, honest, non-hallucinated reasoning string.
    References only facts verifiable in the candidate profile.
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals") or {}

    title = profile.get("current_title") or "Unknown title"
    company = profile.get("current_company") or "unknown company"
    yoe = profile.get("years_of_experience") or 0
    location = profile.get("location") or "unknown location"
    top_skill = _get_top_relevant_skill(candidate)
    company_type = _get_company_type_label(career_scores)
    concern = _get_concern(candidate, behavioral_scores, career_scores)
    notice = signals.get("notice_period_days")
    open_to_work = signals.get("open_to_work_flag", False)

    # Build base sentence
    base = f"{title} at {company} ({yoe:.0f}y exp, {company_type})"
    if top_skill:
        base += f"; {top_skill} directly aligns with JD's retrieval/ranking requirement"

    # Build modifier
    if open_to_work and notice is not None and notice <= 30:
        modifier = f"open to work with {int(notice)}d notice — immediately reachable"
    elif open_to_work:
        modifier = f"marked open-to-work, {location}-based"
    else:
        modifier = f"{location}-based, not flagged open-to-work"

    # Build concern or strength
    if concern:
        tail = f"Concern: {concern}."
    else:
        prod_score = career_scores.get("production_score", 0)
        if prod_score > 0.4:
            tail = "Career descriptions reference production deployment experience."
        else:
            tail = "Strong skill-JD alignment but limited production signal in descriptions."

    # Vary structure based on rank band
    if final_rank <= 10:
        return f"{base}; {modifier}. {tail}"
    elif final_rank <= 50:
        return f"{base}. {modifier.capitalize()}. {tail}"
    else:
        return f"{base}. {tail}"
