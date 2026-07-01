"""
Career Scorer — Lane B (weight 0.35 in final score)

Signals extracted:
  1. Company type score (product vs consulting vs unknown)
  2. Title relevance to ML/AI/Search
  3. Career description keyword density (production-relevant terms)
  4. Seniority band fit (5-9 YOE sweet spot)
  5. Title-chaser detection (too many short tenures)
  6. Recency-weighted scoring (recent roles count more)
  7. Production deployment signal (has candidate shipped real systems?)
"""

from typing import Dict, Any, List
import re

# ─── Company Classification ──────────────────────────────────────────────────

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "l&t infotech", "hexaware", "mindtree", "happiest minds",
    "kpmg", "deloitte", "pwc", "ey", "ernst young",
    "ibm services", "dxc technology"
}

PRODUCT_COMPANY_INDUSTRIES = {
    "saas", "software", "fintech", "edtech", "healthtech", "proptech",
    "ecommerce", "marketplace", "social media", "gaming", "media",
    "ai", "machine learning", "data analytics", "cloud", "cybersecurity",
    "internet", "technology", "startup"
}

# Title relevance tiers (higher = better fit for this JD)
TITLE_TIERS = {
    5: ["ai engineer", "ml engineer", "machine learning engineer", "search engineer",
        "ranking engineer", "recommendation engineer", "nlp engineer",
        "applied scientist", "applied ml", "applied ai", "senior ai",
        "senior ml", "staff ml", "principal ml"],
    4: ["data scientist", "research scientist", "software engineer ml",
        "full stack ml", "platform engineer", "backend engineer",
        "software engineer", "senior software"],
    3: ["data engineer", "analytics engineer", "bi engineer",
        "devops", "sre", "platform", "cloud engineer"],
    2: ["product manager", "program manager", "scrum master",
        "qa engineer", "test engineer"],
    1: ["designer", "business analyst", "operations", "finance",
        "marketing", "sales", "hr", "recruiter", "content", "seo"],
}

# Production keywords from actual career descriptions (not skill tags)
PRODUCTION_KEYWORDS = [
    "production", "deployed", "shipped", "real users", "at scale",
    "serving", "inference", "latency", "throughput", "online",
    "a/b test", "experiment", "canary", "rollout", "monitoring"
]

ML_SYSTEM_KEYWORDS = [
    "embedding", "retrieval", "vector", "similarity", "search",
    "ranking", "recommendation", "rerank", "faiss", "ann",
    "index", "recall", "precision", "ndcg", "encoder",
    "bert", "transformer", "nlp", "information retrieval",
    "rag", "dense", "sparse", "hybrid", "bm25"
]


def _classify_company(company_name: str, industry: str) -> float:
    """
    Returns company type score: 1.0 = product company, 0.3 = consulting, 0.5 = unknown
    """
    cn = (company_name or "").lower()
    ind = (industry or "").lower()

    # Check consulting blacklist
    for firm in CONSULTING_FIRMS:
        if firm in cn:
            return 0.3

    # Check industry for product company signals
    for prod_ind in PRODUCT_COMPANY_INDUSTRIES:
        if prod_ind in ind:
            return 1.0

    # Heuristic: if company name has common SaaS/tech patterns
    tech_patterns = ["tech", "labs", "ai", "data", "cloud", "soft", "systems", "solutions"]
    if any(p in cn for p in tech_patterns):
        return 0.8

    return 0.5  # Unknown


def _get_title_tier(title: str) -> int:
    """Returns 1-5 relevance tier for a job title."""
    t = (title or "").lower()
    for tier, titles in TITLE_TIERS.items():
        for ref in titles:
            if ref in t:
                return tier
    return 2  # Default: "some technical role"


def _keyword_density(text: str, keywords: List[str]) -> float:
    """
    Returns normalized keyword hit rate for a list of keywords in text.
    Score = number of unique hits / len(keywords), capped at 1.0
    """
    text_lower = (text or "").lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return min(hits / max(len(keywords), 1), 1.0)


def _detect_title_chaser(career_history: List[Dict]) -> bool:
    """
    Title chaser: 3+ job changes in 5 years with each < 18 months.
    The JD explicitly says: "optimizing for Senior → Staff → Principal titles by
    switching companies every 1.5 years" is a disqualifier.
    """
    if len(career_history) < 3:
        return False
    short_tenures = [j for j in career_history if 0 < (j.get("duration_months") or 0) < 18]
    return len(short_tenures) >= 3


def _consulting_only(career_history: List[Dict]) -> bool:
    """True if ALL career history is at consulting firms (no product company exposure)."""
    if not career_history:
        return False
    for job in career_history:
        company_score = _classify_company(job.get("company", ""), job.get("industry", ""))
        if company_score > 0.5:  # Found a non-consulting role
            return False
    return True


def score_candidate_career(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Returns a dict of career sub-scores (all 0.0-1.0).
    These are used both for the Lane B composite and as LightGBM features.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history") or []
    yoe = float(profile.get("years_of_experience") or 0)

    if not career:
        return {"career_score": 0.0, "company_score": 0.3, "title_score": 0.3,
                "production_score": 0.0, "ml_system_score": 0.0, "yoe_score": 0.0}

    # ── 1. Company type score (recency-weighted) ─────────────────────────────
    # More recent jobs count more (2x weight for last 2 jobs)
    company_scores = []
    for i, job in enumerate(sorted(career, key=lambda x: x.get("start_date") or "", reverse=True)):
        cs = _classify_company(job.get("company", ""), job.get("industry", ""))
        weight = 2.0 if i < 2 else 1.0
        company_scores.extend([cs] * int(weight))
    company_score = sum(company_scores) / len(company_scores) if company_scores else 0.3

    # Hard penalty: consulting-only with no product company experience
    if _consulting_only(career):
        company_score *= 0.4  # Not 0 — they may have side projects

    # ── 2. Current title relevance ───────────────────────────────────────────
    current_title = profile.get("current_title") or ""
    title_tier = _get_title_tier(current_title)
    title_score = title_tier / 5.0

    # ── 3. Career description quality (production + ML system keywords) ──────
    all_descriptions = " ".join(
        job.get("description") or "" for job in career
    )
    production_score = _keyword_density(all_descriptions, PRODUCTION_KEYWORDS)
    ml_system_score = _keyword_density(all_descriptions, ML_SYSTEM_KEYWORDS)

    # ── 4. YOE band fit ──────────────────────────────────────────────────────
    if 5 <= yoe <= 9:
        yoe_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        yoe_score = 0.8
    elif 3 <= yoe < 4 or 12 < yoe <= 15:
        yoe_score = 0.6
    elif yoe < 2:
        yoe_score = 0.2
    else:
        yoe_score = 0.5  # Very senior (15+ years) — possible but JD is skeptical

    # ── 5. Title-chaser penalty ──────────────────────────────────────────────
    if _detect_title_chaser(career):
        title_score *= 0.6
        company_score *= 0.8

    # ── 6. Composite career score ─────────────────────────────────────────────
    career_score = (
        0.30 * company_score +
        0.25 * title_score +
        0.20 * ml_system_score +
        0.15 * production_score +
        0.10 * yoe_score
    )

    return {
        "career_score": career_score,
        "company_score": company_score,
        "title_score": title_score,
        "production_score": production_score,
        "ml_system_score": ml_system_score,
        "yoe_score": yoe_score,
    }
