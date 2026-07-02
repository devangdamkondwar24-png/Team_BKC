"""
Feature Extractor — Structured fit scoring (0.25 weight).
YoE, skill overlap, education, location, notice period, tenure discipline.
"""
from src.config import (
    MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, CORE_AI_SKILLS,
    PREFERRED_LOCATIONS, ACCEPTABLE_LOCATIONS,
    YOE_IDEAL_MIN, YOE_IDEAL_MAX,
)


def _yoe_score(candidate: dict) -> float:
    """Sweet spot 5-9 years → 1.0; decay outside."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if YOE_IDEAL_MIN <= yoe <= YOE_IDEAL_MAX:
        return 1.0
    elif 4 <= yoe < YOE_IDEAL_MIN:
        return 0.8
    elif YOE_IDEAL_MAX < yoe <= 12:
        return 0.8 - 0.1 * (yoe - YOE_IDEAL_MAX)
    elif yoe < 3:
        return max(0.1, 0.3 * yoe / 3)
    else:  # >12
        return max(0.3, 0.8 - 0.05 * (yoe - 12))


def _skill_overlap_must_have(candidate: dict) -> float:
    """Count intersection with must-have skill families. Each family → +0.25."""
    cand_skills = {s.get("name", "").lower() for s in candidate.get("skills", [])}
    # Also check career descriptions for implicit skill mentions
    all_desc = " ".join(
        j.get("description", "") for j in candidate.get("career_history", [])
    ).lower()

    families_matched = 0
    for family_name, family_skills in MUST_HAVE_SKILLS.items():
        family_lower = {s.lower() for s in family_skills}
        # Check both skill list and career descriptions
        if cand_skills & family_lower:
            families_matched += 1
        elif any(s in all_desc for s in family_lower):
            families_matched += 0.7  # partial credit for description mentions
    return min(1.0, families_matched / len(MUST_HAVE_SKILLS))


def _skill_overlap_nice_to_have(candidate: dict) -> float:
    """Count nice-to-have skill families present."""
    cand_skills = {s.get("name", "").lower() for s in candidate.get("skills", [])}
    all_desc = " ".join(
        j.get("description", "") for j in candidate.get("career_history", [])
    ).lower()

    families_matched = 0
    for family_name, family_skills in NICE_TO_HAVE_SKILLS.items():
        family_lower = {s.lower() for s in family_skills}
        if cand_skills & family_lower:
            families_matched += 1
        elif any(s in all_desc for s in family_lower):
            families_matched += 0.5
    return min(1.0, families_matched / len(NICE_TO_HAVE_SKILLS))


def _education_score(candidate: dict) -> float:
    """Education tier + field relevance."""
    edu_list = candidate.get("education", [])
    if not edu_list:
        return 0.2
    best = 0.0
    for edu in edu_list:
        tier = edu.get("tier", "unknown")
        tier_map = {"tier_1": 1.0, "tier_2": 0.8, "tier_3": 0.5, "tier_4": 0.3, "unknown": 0.4}
        tier_score = tier_map.get(tier, 0.3)

        field = edu.get("field_of_study", "").lower()
        cs_fields = {"computer science", "machine learning", "artificial intelligence",
                     "data science", "information technology", "software engineering",
                     "electronics", "electrical engineering", "mathematics", "statistics"}
        field_bonus = 0.2 if any(f in field for f in cs_fields) else 0.0

        degree = edu.get("degree", "").lower()
        degree_bonus = 0.1 if degree in ("m.tech", "m.sc", "ph.d", "phd", "m.e.") else 0.0

        best = max(best, tier_score + field_bonus + degree_bonus)
    return min(1.0, best)


def _location_score(candidate: dict) -> float:
    """Location match per JD preferences."""
    profile = candidate.get("profile", {})
    location = profile.get("location", "")
    country = profile.get("country", "")
    willing = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    loc_lower = location.lower()

    # Check preferred cities
    if any(c.lower() in loc_lower for c in PREFERRED_LOCATIONS):
        return 1.0
    # Check acceptable cities
    if any(c.lower() in loc_lower for c in ACCEPTABLE_LOCATIONS):
        return 0.8
    # Other India locations
    if country == "India":
        return 0.7 if willing else 0.4
    # Outside India
    return 0.15 if willing else 0.05


def _notice_period_score(candidate: dict) -> float:
    """JD: sub-30-day preferred, 30+ still in scope but higher bar."""
    notice = candidate.get("redrob_signals", {}).get("notice_period_days", 90)
    if notice <= 30:
        return 1.0
    elif notice <= 60:
        return 0.7
    elif notice <= 90:
        return 0.4
    return 0.2


def _tenure_discipline_score(candidate: dict) -> float:
    """JD: 'not a 1.5-year job-hopper chasing titles'. Score avg tenure."""
    career = candidate.get("career_history", [])
    if len(career) <= 1:
        return 0.7  # Single job — neutral
    durations = [j.get("duration_months", 0) for j in career]
    avg = sum(durations) / len(durations) if durations else 0
    if avg >= 36:
        return 1.0
    elif avg >= 24:
        return 0.8
    elif avg >= 18:
        return 0.5
    return 0.2


def structured_fit_score(candidate: dict) -> float:
    """Composite structured fit score (0-1).
    Sub-weights: YoE 0.15, must-have skills 0.25, nice-to-have 0.10,
    education 0.10, location 0.20, notice 0.10, tenure 0.10."""
    return min(1.0, max(0.0,
        0.15 * _yoe_score(candidate)
        + 0.25 * _skill_overlap_must_have(candidate)
        + 0.10 * _skill_overlap_nice_to_have(candidate)
        + 0.10 * _education_score(candidate)
        + 0.20 * _location_score(candidate)
        + 0.10 * _notice_period_score(candidate)
        + 0.10 * _tenure_discipline_score(candidate)
    ))
