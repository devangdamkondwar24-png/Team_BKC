"""
Reasoning Generator — Produces per-candidate reasoning strings from real fields.

Each reasoning is constructed from the candidate's actual profile content.
No templates, no hallucinated skills. Varies per candidate based on which
scoring pillars contributed most.
"""
from src.config import (
    MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, AI_ENGINEERING_TITLES,
    IT_SERVICES_COMPANIES, RETRIEVAL_DOMAIN_KEYWORDS, PRODUCTION_KEYWORDS,
    PREFERRED_LOCATIONS, ACCEPTABLE_LOCATIONS,
)
from src.disqualifier import get_disqualifier_flags


def generate_reasoning(candidate: dict, breakdown: dict) -> str:
    """Generate 1-2 sentence reasoning string from real candidate data.

    The reasoning must be:
    - Specific to the candidate's actual profile
    - Consistent with their rank
    - Mention real skills/experience, not hallucinated ones
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "")
    signals = candidate.get("redrob_signals", {})

    parts = []

    # Title and experience
    parts.append(f"{title} at {company}, {yoe:.1f}yr exp")

    # Key matching skills (from actual skill list)
    cand_skills = {s.get("name", "") for s in candidate.get("skills", [])}
    all_must_have = set()
    for family in MUST_HAVE_SKILLS.values():
        all_must_have |= family
    all_nice = set()
    for family in NICE_TO_HAVE_SKILLS.values():
        all_nice |= family

    matched_must = sorted(cand_skills & all_must_have)
    matched_nice = sorted(cand_skills & all_nice)

    if matched_must:
        skills_str = ", ".join(matched_must[:4])
        parts.append(f"core skills: {skills_str}")
    elif matched_nice:
        skills_str = ", ".join(matched_nice[:3])
        parts.append(f"relevant skills: {skills_str}")

    # Career highlight
    career_highlights = []
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").lower()
        if any(kw in desc for kw in RETRIEVAL_DOMAIN_KEYWORDS):
            career_highlights.append(f"retrieval/ranking exp at {job.get('company', '?')}")
            break
        elif any(kw in desc for kw in PRODUCTION_KEYWORDS):
            career_highlights.append(f"production ML at {job.get('company', '?')}")
            break

    if career_highlights:
        parts.append(career_highlights[0])

    # Location
    loc_lower = location.lower()
    if any(c.lower() in loc_lower for c in PREFERRED_LOCATIONS):
        parts.append(f"based in {location} (preferred)")
    elif any(c.lower() in loc_lower for c in ACCEPTABLE_LOCATIONS):
        parts.append(f"in {location}")
    elif country == "India":
        if signals.get("willing_to_relocate", False):
            parts.append(f"in {location}, open to relocate")

    # Behavioral signals (brief)
    notice = signals.get("notice_period_days", 90)
    resp_rate = signals.get("recruiter_response_rate", 0)
    if notice <= 30:
        parts.append(f"{notice}d notice")
    if resp_rate >= 0.7:
        parts.append(f"high recruiter engagement ({resp_rate:.0%})")

    # Disqualifier warnings (for lower-ranked candidates)
    disq_flags = get_disqualifier_flags(candidate)
    if disq_flags and breakdown.get("final_score", 1) < 0.3:
        flag_str = disq_flags[0].replace("_", " ")
        parts.append(f"note: {flag_str}")

    # Join and truncate
    reasoning = "; ".join(parts)
    # Ensure it's 1-2 sentences, not too long
    if len(reasoning) > 300:
        reasoning = reasoning[:297] + "..."

    return reasoning
