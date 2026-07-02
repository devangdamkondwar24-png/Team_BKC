"""
Behavioral Scorer — Redrob signals composite (0.10 weight).
Measures availability, reachability, and platform engagement.
"""
from datetime import datetime
from src.config import (
    REFERENCE_DATE,
    LAST_ACTIVE_RECENT_DAYS, LAST_ACTIVE_MODERATE_DAYS, LAST_ACTIVE_STALE_DAYS,
    RESPONSE_TIME_FAST_HOURS, RESPONSE_TIME_MODERATE_HOURS, RESPONSE_TIME_SLOW_HOURS,
)


def behavioral_score(candidate: dict) -> float:
    """Composite behavioral/availability score (0-1).

    Sub-weights: open_to_work 0.15, recency 0.20, response_rate 0.20,
    response_time 0.10, github 0.10, interview_completion 0.10,
    profile_completeness 0.05, verification 0.10.
    """
    signals = candidate.get("redrob_signals", {})

    # Open to work flag
    otw = 1.0 if signals.get("open_to_work_flag", False) else 0.3

    # Last active recency
    last_active_str = signals.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d")
        days_since = (REFERENCE_DATE - last_active).days
        if days_since <= LAST_ACTIVE_RECENT_DAYS:
            recency = 1.0
        elif days_since <= LAST_ACTIVE_MODERATE_DAYS:
            recency = 0.7
        elif days_since <= LAST_ACTIVE_STALE_DAYS:
            recency = 0.4
        else:
            recency = 0.1
    except (ValueError, TypeError):
        recency = 0.3

    # Recruiter response rate (0-1 directly)
    response_rate = signals.get("recruiter_response_rate", 0.0)

    # Response time
    resp_time = signals.get("avg_response_time_hours", 200)
    if resp_time <= RESPONSE_TIME_FAST_HOURS:
        resp_time_score = 1.0
    elif resp_time <= RESPONSE_TIME_MODERATE_HOURS:
        resp_time_score = 0.7
    elif resp_time <= RESPONSE_TIME_SLOW_HOURS:
        resp_time_score = 0.4
    else:
        resp_time_score = 0.1

    # GitHub activity
    github = signals.get("github_activity_score", -1)
    if github < 0:
        github_score = 0.2  # Unknown, slight penalty
    else:
        github_score = github / 100.0

    # Interview completion rate
    interview = signals.get("interview_completion_rate", 0.0)

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 50) / 100.0

    # Verification flags
    verified = (
        (1 if signals.get("verified_email", False) else 0)
        + (1 if signals.get("verified_phone", False) else 0)
        + (1 if signals.get("linkedin_connected", False) else 0)
    ) / 3.0

    score = (
        0.15 * otw
        + 0.20 * recency
        + 0.20 * response_rate
        + 0.10 * resp_time_score
        + 0.10 * github_score
        + 0.10 * interview
        + 0.05 * completeness
        + 0.10 * verified
    )
    return min(1.0, max(0.0, score))
