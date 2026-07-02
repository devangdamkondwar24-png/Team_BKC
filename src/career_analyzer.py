"""
Career Analyzer — Production AI experience depth scoring.
Highest-weighted pillar (0.40) — the JD's primary differentiator.
"""
import re
from src.config import (
    AI_ENGINEERING_TITLES, SOFTWARE_ENGINEERING_TITLES, NON_TECH_TITLES,
    IT_SERVICES_COMPANIES, PRODUCTION_KEYWORDS, RETRIEVAL_DOMAIN_KEYWORDS,
)


def _title_seniority_score(candidate: dict) -> float:
    """Map current title to engineering relevance. AI/ML→1.0, SWE→0.5, non-tech→0.02."""
    title = candidate.get("profile", {}).get("current_title", "")
    if title in AI_ENGINEERING_TITLES:
        tl = title.lower()
        if any(w in tl for w in ("senior", "staff", "lead", "principal")):
            return 1.0
        elif "junior" in tl:
            return 0.6
        return 0.85
    if title in SOFTWARE_ENGINEERING_TITLES:
        tl = title.lower()
        if "senior" in tl:
            return 0.7
        if any(w in tl for w in ("data engineer", "analytics", "backend")):
            return 0.55
        return 0.45
    if title in NON_TECH_TITLES:
        return 0.02
    return 0.3


def _production_experience_score(candidate: dict) -> float:
    """Scan career descriptions for production deployment signals, weighted by months."""
    total_prod_months = 0
    total_months = 0
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur
        prod_hits = sum(1 for kw in PRODUCTION_KEYWORDS if kw in desc)
        if prod_hits >= 2:
            total_prod_months += dur
        elif prod_hits == 1:
            total_prod_months += dur * 0.5
    if total_months == 0:
        return 0.0
    return min(1.0, (total_prod_months / total_months) * 1.5)


def _product_vs_services_score(candidate: dict) -> float:
    """Penalize career entirely at IT services firms."""
    career = candidate.get("career_history", [])
    if not career:
        return 0.3
    services_months = 0
    total_months = 0
    for job in career:
        dur = job.get("duration_months", 0)
        total_months += dur
        if job.get("company", "") in IT_SERVICES_COMPANIES:
            services_months += dur
        elif job.get("industry", "") == "IT Services":
            services_months += dur * 0.7
    if total_months == 0:
        return 0.3
    ratio = services_months / total_months
    if ratio >= 0.95:
        return 0.1
    elif ratio >= 0.7:
        return 0.35
    elif ratio >= 0.4:
        return 0.6
    elif ratio >= 0.1:
        return 0.8
    return 1.0


def _retrieval_domain_score(candidate: dict) -> float:
    """Score for ranking/retrieval/search/recommendation domain experience."""
    total_hits = 0
    total_months_with_hits = 0
    total_months = 0
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur
        hits = sum(1 for kw in RETRIEVAL_DOMAIN_KEYWORDS if kw in desc)
        if hits >= 2:
            total_months_with_hits += dur
            total_hits += hits
        elif hits == 1:
            total_months_with_hits += dur * 0.5
            total_hits += hits
    if total_months == 0:
        return 0.0
    months_score = min(1.0, total_months_with_hits / max(total_months, 1) * 2.0)
    breadth_score = min(1.0, total_hits / 8.0)
    return 0.6 * months_score + 0.4 * breadth_score


def career_depth_score(candidate: dict) -> float:
    """Composite career depth score (0-1). Sub-weights: title 0.25, production 0.30,
    company type 0.20, retrieval domain 0.25."""
    return min(1.0, max(0.0,
        0.25 * _title_seniority_score(candidate)
        + 0.30 * _production_experience_score(candidate)
        + 0.20 * _product_vs_services_score(candidate)
        + 0.25 * _retrieval_domain_score(candidate)
    ))
