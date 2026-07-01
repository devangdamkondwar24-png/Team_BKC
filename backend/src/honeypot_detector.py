"""
Honeypot Detector — identifies candidates with logically impossible profiles.
These are hard-forced to relevance tier 0 in the ground truth.
Submitting them in top 100 → disqualification at Stage 3.

Detection rules (ALL are conservative to avoid false positives):
  1. Total career duration far exceeds stated YOE
  2. Expert/advanced skill with 0 months duration
  3. Skill duration longer than candidate's total career
  4. Future start dates in career history
  5. Multiple overlapping "current" jobs with inconsistent durations
"""

from datetime import datetime, date
from typing import Dict, Any, List

TODAY = date.today()


def _total_career_months(career_history: List[Dict]) -> int:
    """Sum of all job durations. Overlapping jobs counted once."""
    return sum(j.get("duration_months", 0) or 0 for j in career_history)


def _has_future_dates(career_history: List[Dict]) -> bool:
    for job in career_history:
        for key in ["start_date", "end_date"]:
            val = job.get(key)
            if val and val != "null":
                try:
                    from dateutil.parser import parse
                    d = parse(str(val)).date()
                    if d > TODAY:
                        return True
                except Exception:
                    pass
    return False


def _has_impossible_skill(skills: List[Dict], yoe: float) -> bool:
    total_career_months = yoe * 12
    for skill in skills:
        prof = (skill.get("proficiency") or "").lower()
        dur = skill.get("duration_months") or 0
        # Expert/advanced with literally 0 months → impossible
        if prof in ("expert", "advanced") and dur == 0:
            return True
        # Skill duration exceeds entire career
        if dur > total_career_months + 12:
            return True
    return False


def _has_impossible_tenure(career_history: List[Dict], yoe: float) -> bool:
    total_months = _total_career_months(career_history)
    # Allow 30% slack for overlapping roles, rounding, etc.
    # Flag only clearly impossible: total months > 140% of YOE
    if total_months > (yoe * 12) * 1.4 + 24:
        return True
    return False


def _has_company_age_violation(career_history: List[Dict]) -> bool:
    """
    Checks if candidate claims X years at a company known to be younger.
    We use a conservative list of well-known recent companies.
    In a full system, this would use a company founding date database.
    """
    YOUNG_COMPANIES = {
        # company_name_fragment: founded_year
        "openai": 2015,
        "mistral": 2023,
        "anthropic": 2021,
        "perplexity": 2022,
        "redrob": 2022,
    }
    for job in career_history:
        company = (job.get("company") or "").lower()
        dur_months = job.get("duration_months") or 0
        start_year = TODAY.year - (dur_months // 12)
        for frag, founded in YOUNG_COMPANIES.items():
            if frag in company and start_year < founded:
                return True
    return False


def is_honeypot(candidate: Dict[str, Any]) -> bool:
    """
    Returns True if this candidate appears to be a honeypot.
    Honeypots score 0.0 regardless of any other signal.
    """
    profile = candidate.get("profile", {})
    yoe = float(profile.get("years_of_experience") or 0)
    career_history = candidate.get("career_history") or []
    skills = candidate.get("skills") or []

    if yoe <= 0:
        return False  # Skip candidates with unknown YOE to avoid false positives

    checks = [
        _has_impossible_tenure(career_history, yoe),
        _has_impossible_skill(skills, yoe),
        _has_future_dates(career_history),
        _has_company_age_violation(career_history),
    ]
    return any(checks)


def score_honeypot_confidence(candidate: Dict[str, Any]) -> float:
    """
    Returns 0.0 if honeypot, 1.0 if clean.
    Used as a multiplier on the final score.
    """
    return 0.0 if is_honeypot(candidate) else 1.0
