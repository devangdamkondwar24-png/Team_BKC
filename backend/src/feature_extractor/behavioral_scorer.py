"""
Behavioral Scorer — Lane C (weight 0.20 in final score)

Converts redrob_signals into an "availability multiplier".
This is what separates a theoretical fit from an actual hire.

Key principle: behavioral signals are MULTIPLIERS, not additive bonuses.
A great technical fit with terrible availability signals → still bad for a recruiter.
"""

from typing import Dict, Any
from datetime import date, datetime
from dateutil.parser import parse as parse_date


TODAY = date.today()


def _days_since(date_str: str) -> int:
    """Returns days since a date string. Returns 999 if invalid."""
    if not date_str:
        return 999
    try:
        d = parse_date(str(date_str)).date()
        return (TODAY - d).days
    except Exception:
        return 999


def _recency_score(last_active_date: str) -> float:
    """
    How recently was the candidate active on the platform?
    Recency is the #1 availability signal — an inactive profile is an unavailable candidate.
    """
    days = _days_since(last_active_date)
    if days <= 7:    return 1.0
    if days <= 30:   return 0.95
    if days <= 60:   return 0.85
    if days <= 90:   return 0.70
    if days <= 180:  return 0.50
    if days <= 365:  return 0.25
    return 0.10  # > 1 year inactive: extremely unlikely to respond


def _notice_penalty(notice_days: float) -> float:
    """
    JD wants sub-30 day notice. Can buy out up to 30 days.
    30+ day candidates: 'bar gets higher' per JD.
    """
    if notice_days is None or notice_days < 0:
        return 0.7  # Unknown notice — moderate penalty
    if notice_days <= 15:   return 1.0
    if notice_days <= 30:   return 0.95
    if notice_days <= 60:   return 0.80
    if notice_days <= 90:   return 0.65
    if notice_days <= 120:  return 0.50
    if notice_days <= 150:  return 0.40
    return 0.30  # 150+ days: extremely unlikely to join quickly


def _location_score(candidate: Dict[str, Any], signals: Dict[str, Any]) -> float:
    """
    Location scoring: Pune/Noida preferred. India Tier-1 cities welcome.
    Willing to relocate from anywhere in India is acceptable.
    Outside India: case-by-case (JD says so).
    """
    TIER1_INDIA = {
        "pune", "noida", "hyderabad", "mumbai", "delhi", "bengaluru",
        "bangalore", "gurugram", "gurgaon", "chennai", "ncr"
    }
    PREFERRED = {"pune", "noida"}

    profile = candidate.get("profile", {})
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing_to_relocate = signals.get("willing_to_relocate", False)

    # Pune/Noida → best
    if any(p in location for p in PREFERRED):
        return 1.0

    # Other Tier-1 India cities → very good
    if any(city in location for city in TIER1_INDIA) or country == "india":
        if willing_to_relocate:
            return 0.95
        return 0.88

    # Willing to relocate from anywhere in India → acceptable
    if country == "india" and willing_to_relocate:
        return 0.85

    # Outside India → risky (JD: case-by-case, no visa sponsorship)
    if willing_to_relocate:
        return 0.60
    return 0.40


def score_candidate_behavioral(candidate: Dict[str, Any]) -> Dict[str, float]:
    """
    Returns a dict of behavioral sub-scores (all 0.0-1.0).
    """
    signals = candidate.get("redrob_signals") or {}

    # ── 1. Recency (most important behavioral signal) ─────────────────────────
    recency = _recency_score(signals.get("last_active_date", ""))

    # ── 2. Open-to-work flag ──────────────────────────────────────────────────
    open_to_work = 1.0 if signals.get("open_to_work_flag") else 0.75

    # ── 3. Recruiter response rate ────────────────────────────────────────────
    rr = signals.get("recruiter_response_rate")
    if rr is None:
        response_score = 0.6
    elif rr >= 0.7:
        response_score = 1.0
    elif rr >= 0.4:
        response_score = 0.85
    elif rr >= 0.2:
        response_score = 0.65
    elif rr >= 0.1:
        response_score = 0.45
    else:
        response_score = 0.20  # < 10% response rate → not reachable

    # ── 4. Notice period ──────────────────────────────────────────────────────
    notice_score = _notice_penalty(signals.get("notice_period_days"))

    # ── 5. Interview completion rate ──────────────────────────────────────────
    icr = signals.get("interview_completion_rate")
    if icr is None:
        interview_score = 0.7
    elif icr >= 0.8:
        interview_score = 1.0
    elif icr >= 0.6:
        interview_score = 0.85
    elif icr >= 0.4:
        interview_score = 0.65
    else:
        interview_score = 0.40  # Ghosts interviews frequently

    # ── 6. Platform engagement (saved by recruiters, profile completeness) ────
    saved_30d = min((signals.get("saved_by_recruiters_30d") or 0) / 10.0, 1.0)
    completeness = (signals.get("profile_completeness_score") or 0) / 100.0
    github_score = max((signals.get("github_activity_score") or -1), 0) / 100.0

    # ── 7. Location score ─────────────────────────────────────────────────────
    location_score = _location_score(candidate, signals)

    # ── 8. Work mode alignment ────────────────────────────────────────────────
    work_mode = (signals.get("preferred_work_mode") or "").lower()
    # JD says hybrid (Tue/Thu in office). Remote-only is a misalignment.
    if work_mode in ("hybrid", "flexible", "onsite"):
        work_mode_score = 1.0
    elif work_mode == "remote":
        work_mode_score = 0.70  # Not disqualifying, but suboptimal
    else:
        work_mode_score = 0.85  # Unknown

    # ── 9. Composite behavioral score ─────────────────────────────────────────
    behavioral_score = (
        0.30 * recency +
        0.20 * response_score +
        0.15 * notice_score +
        0.10 * open_to_work +
        0.10 * location_score +
        0.08 * interview_score +
        0.04 * completeness +
        0.03 * github_score
    )

    return {
        "behavioral_score": behavioral_score,
        "recency_score": recency,
        "response_score": response_score,
        "notice_score": notice_score,
        "location_score": location_score,
        "open_to_work": float(open_to_work),
        "interview_score": interview_score,
        "github_score": github_score,
        "completeness_score": completeness,
        "saved_score": saved_30d,
    }
