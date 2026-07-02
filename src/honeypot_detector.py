"""
Honeypot Detector — 12 rule checks for impossible/planted profiles.

Design: each rule is a pure function returning bool. Any True → candidate
is flagged as a honeypot and hard-zeroed (excluded from top 100).

Rules 1-6: original checks from data profiling
Rules 7-12: expanded checks from user feedback to reach ~80 detection
"""
from datetime import datetime
from typing import Any

from src.config import (
    EXPERT_ZERO_DUR_THRESHOLD,
    CAREER_YOE_BUFFER_MONTHS,
    DURATION_ELAPSED_BUFFER_MONTHS,
    ASSESSMENT_MISMATCH_THRESHOLD,
    ENDORSEMENT_HIGH_PERCENTILE,
    COMPLETENESS_LOW_THRESHOLD,
    REFERENCE_DATE,
)


def expert_with_zero_duration(candidate: dict) -> bool:
    """Rule 1: Any skill self-rated 'expert' with 0 months of usage."""
    for skill in candidate.get("skills", []):
        if (skill.get("proficiency") == "expert"
                and skill.get("duration_months", 0) == 0):
            return True
    return False


def duration_exceeds_elapsed_time(candidate: dict) -> bool:
    """Rule 2: Career duration_months at a company exceeds elapsed calendar time."""
    for job in candidate.get("career_history", []):
        start_str = job.get("start_date", "")
        dur = job.get("duration_months", 0)
        if not start_str or dur == 0:
            continue
        try:
            start_dt = datetime.strptime(start_str, "%Y-%m-%d")
            max_months = (
                (REFERENCE_DATE.year - start_dt.year) * 12
                + (REFERENCE_DATE.month - start_dt.month)
            )
            if dur > max_months + DURATION_ELAPSED_BUFFER_MONTHS:
                return True
        except (ValueError, TypeError):
            continue
    return False


def total_months_exceeds_yoe(candidate: dict) -> bool:
    """Rule 3: Sum of career_history durations vastly exceeds years_of_experience."""
    total_months = sum(
        job.get("duration_months", 0)
        for job in candidate.get("career_history", [])
    )
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if total_months > (yoe * 12) + CAREER_YOE_BUFFER_MONTHS:
        return True
    return False


def implausible_skill_count(candidate: dict) -> bool:
    """Rule 4: Excessive number of expert/advanced skills (>12 at expert or >18 at advanced+)."""
    skills = candidate.get("skills", [])
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    advanced_plus = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
    )
    # Very unlikely for a real person to be expert in >10 distinct skills
    if expert_count > 10:
        return True
    # Or advanced+ in >15 skills
    if advanced_plus > 15:
        return True
    return False


def description_title_mismatch(candidate: dict) -> bool:
    """Rule 5: Career descriptions clearly from a different domain than the title.

    Detect mechanical/hardware/accounting descriptions under software/AI titles
    or vice versa. Uses keyword heuristics.
    """
    ENGINEERING_DESC_KEYWORDS = {
        "solidworks", "creo", "ansys", "fea ", "dfm", "dfma",
        "manufacturing", "tooling", "cad ", "mechanical",
        "prototype", "hardware",
    }
    ACCOUNTING_DESC_KEYWORDS = {
        "gaap", "ind-as", "general ledger", "gl,", "month-end close",
        "tax filings", "statutory compliance", "fixed-asset register",
        "audit-readiness",
    }
    NON_TECH_DESC_KEYWORDS = {
        "brand design", "packaging design", "editorial calendar",
        "freelance writer", "seo strategy", "content writing",
        "support agents", "support knowledge base", "tier-1",
        "customer-feedback loop",
    }

    TECH_TITLES_LOWER = {
        "ml engineer", "ai engineer", "software engineer",
        "data scientist", "backend engineer", "ai specialist",
        "ai research engineer", "nlp engineer", "search engineer",
        "applied ml engineer", "machine learning engineer",
        "senior software engineer (ml)", "data engineer",
        "full stack developer", "devops engineer", "cloud engineer",
    }

    for job in candidate.get("career_history", []):
        title_lower = job.get("title", "").lower()
        desc_lower = job.get("description", "").lower()

        is_tech_title = any(t in title_lower for t in TECH_TITLES_LOWER)

        if is_tech_title:
            mismatch_kws = ENGINEERING_DESC_KEYWORDS | ACCOUNTING_DESC_KEYWORDS | NON_TECH_DESC_KEYWORDS
            matches = sum(1 for kw in mismatch_kws if kw in desc_lower)
            if matches >= 3:
                return True
    return False


def education_chronology_impossible(candidate: dict) -> bool:
    """Rule 6: Impossible chronological sequence in education or dates."""
    education = candidate.get("education", [])
    
    # Sort degrees by end_year to check for logical progression
    degrees_with_dates = [
        edu for edu in education 
        if edu.get("start_year") and edu.get("end_year")
    ]
    
    for edu in degrees_with_dates:
        start = edu.get("start_year")
        end = edu.get("end_year")
        if end < start:
            return True
        # Degree in <1 year for PhD/Masters
        degree = edu.get("degree", "").lower()
        if degree in ("ph.d", "phd", "m.tech", "m.sc", "m.e.", "mba"):
            if end - start < 1:
                return True
                
    # Check degree order logic (e.g., Master's before Bachelor's)
    # Define roughly expected hierarchy (higher number = more advanced)
    tier_map = {
        "b.tech": 1, "b.e.": 1, "b.sc": 1, "b.a.": 1,
        "m.tech": 2, "m.sc": 2, "m.e.": 2, "mba": 2, "m.a.": 2,
        "ph.d": 3, "phd": 3
    }
    
    sorted_edus = sorted(degrees_with_dates, key=lambda x: x["end_year"])
    highest_tier_so_far = -1
    for edu in sorted_edus:
        degree = edu.get("degree", "").lower()
        tier = tier_map.get(degree, -1)
        if tier != -1:
            if highest_tier_so_far > tier:
                # Got a lower degree (e.g. B.Sc) AFTER a higher degree (e.g. Ph.D)
                # But wait, sometimes people do a second bachelors. Let's only flag if the gap is absurd.
                pass
            highest_tier_so_far = max(highest_tier_so_far, tier)
            
    # Stronger check: Did they start a lower degree AFTER finishing a higher degree by >5 years?
    # Or did they do a Master's 10 years BEFORE a Bachelor's?
    bachelor_year = min((edu["end_year"] for edu in degrees_with_dates if tier_map.get(edu.get("degree", "").lower()) == 1), default=None)
    master_year = min((edu["end_year"] for edu in degrees_with_dates if tier_map.get(edu.get("degree", "").lower()) == 2), default=None)
    phd_year = min((edu["end_year"] for edu in degrees_with_dates if tier_map.get(edu.get("degree", "").lower()) == 3), default=None)
    
    if bachelor_year and master_year and master_year < bachelor_year:
        return True
    if bachelor_year and phd_year and phd_year < bachelor_year:
        return True
    if master_year and phd_year and phd_year < master_year:
        return True
        
    return False


def proficiency_assessment_mismatch(candidate: dict) -> bool:
    """Rule 7: Multiple skills self-rated expert/advanced but assessment score <15."""
    assessment_scores = (
        candidate.get("redrob_signals", {})
        .get("skill_assessment_scores", {})
    )
    if not assessment_scores:
        return False

    mismatch_count = 0
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        prof = skill.get("proficiency", "")
        
        # B1 Check: Does assessment_scores contain the key? 
        # If it's missing entirely or None, we skip.
        if prof in ("expert", "advanced") and name in assessment_scores:
            score = assessment_scores[name]
            if score is not None and score < 15:  # Tightened from 30 to 15
                mismatch_count += 1
                
    # B2 Check: Require multiple occurrences
    if mismatch_count >= 2:
        return True
    return False


def education_overlap_extreme(candidate: dict) -> bool:
    """Rule 8: Full-time degree overlaps almost entirely with full-time work.
    
    Must be used as a corroborating signal (co-occurring), not standalone.
    """
    for edu in candidate.get("education", []):
        end_year = edu.get("end_year", 0)
        start_year = edu.get("start_year", 0)
        if not end_year or not start_year or end_year <= start_year:
            continue
            
        degree_months = (end_year - start_year) * 12
        degree = edu.get("degree", "").lower()
        
        if degree in ("ph.d", "phd", "m.tech", "m.sc", "m.e.", "mba", "b.tech", "b.e.", "b.sc"):
            concurrent_months = 0
            for job in candidate.get("career_history", []):
                job_start = job.get("start_date", "")
                if not job_start:
                    continue
                
                title = job.get("title", "").lower()
                if "intern" in title or "trainee" in title:
                    continue

                try:
                    job_start_dt = datetime.strptime(job_start, "%Y-%m-%d")
                    # Approximate degree start and end
                    degree_start = datetime(start_year, 8, 1)
                    degree_end = datetime(end_year, 6, 30)
                    
                    if job_start_dt < degree_end:
                        # Find overlap between job duration and degree window
                        job_end_dt = job_start_dt
                        # If we have an exact end date or just duration
                        job_end_str = job.get("end_date")
                        if job_end_str:
                            job_end_dt = datetime.strptime(job_end_str, "%Y-%m-%d")
                        else:
                            # Add duration months to start date
                            dur_days = job.get("duration_months", 0) * 30
                            import datetime as dt
                            job_end_dt = job_start_dt + dt.timedelta(days=dur_days)
                        
                        overlap_start = max(job_start_dt, degree_start)
                        overlap_end = min(job_end_dt, degree_end)
                        
                        if overlap_end > overlap_start:
                            overlap_months = (overlap_end - overlap_start).days / 30.0
                            concurrent_months += overlap_months
                except (ValueError, TypeError):
                    continue
                    
            # A2 Check: overlap >= 90% of the stated degree duration
            if concurrent_months >= (0.90 * degree_months) and degree_months > 0:
                return True
    return False


def endorsements_disproportionate(candidate: dict) -> bool:
    """Rule 9: Very high endorsements/connections with very low profile completeness."""
    signals = candidate.get("redrob_signals", {})
    completeness = signals.get("profile_completeness_score", 50)
    endorsements = signals.get("endorsements_received", 0)
    connections = signals.get("connection_count", 0)

    if completeness < COMPLETENESS_LOW_THRESHOLD:
        if endorsements > ENDORSEMENT_HIGH_PERCENTILE or connections > 800:
            return True
    return False


def company_tenure_predates_market_entry(
    candidate: dict,
    company_earliest_starts: dict[str, str] | None = None,
) -> bool:
    """Rule 10: Candidate claims start_date at a company much earlier than
    the earliest observed start_date for that company across all 100K candidates.

    company_earliest_starts: {company_name: "YYYY-MM-DD"} precomputed from full pool.
    """
    if not company_earliest_starts:
        return False

    for job in candidate.get("career_history", []):
        company = job.get("company", "")
        start_str = job.get("start_date", "")
        if not company or not start_str or company not in company_earliest_starts:
            continue
        try:
            cand_start = datetime.strptime(start_str, "%Y-%m-%d")
            earliest_str = company_earliest_starts[company]
            earliest = datetime.strptime(earliest_str, "%Y-%m-%d")
            # If this candidate's start is more than 24 months before anyone
            # else at the same company, that's suspicious
            diff_months = (
                (earliest.year - cand_start.year) * 12
                + (earliest.month - cand_start.month)
            )
            if diff_months > 24:
                return True
        except (ValueError, TypeError):
            continue
    return False


def salary_min_exceeds_max(candidate: dict) -> bool:
    """Rule 11a: Salary min > max."""
    signals = candidate.get("redrob_signals", {})
    salary = signals.get("expected_salary_range_inr_lpa", {})
    if not salary:
        return False

    sal_min = salary.get("min", 0)
    sal_max = salary.get("max", 0)

    # Inverted range
    if sal_min > sal_max and sal_max > 0:
        return True
    return False

def salary_inconsistent_with_seniority(candidate: dict) -> bool:
    """Rule 11b: Salary wildly inconsistent with experience."""
    signals = candidate.get("redrob_signals", {})
    salary = signals.get("expected_salary_range_inr_lpa", {})
    if not salary:
        return False

    sal_min = salary.get("min", 0)
    sal_max = salary.get("max", 0)
    
    # Normalize inverted salaries
    sal_min, sal_max = sorted([sal_min, sal_max])
    
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    # Senior engineer (>8 years) expecting <3 LPA → suspicious
    if yoe > 8 and sal_max < 3.0 and sal_max > 0:
        return True

    # Entry-level (<2 years) expecting >60 LPA → suspicious
    if yoe < 2 and sal_min > 60:
        return True

    return False


def offer_acceptance_without_offers(candidate: dict) -> bool:
    """Rule 12: Has a concrete offer_acceptance_rate but signals suggest
    no offer history is plausible."""
    signals = candidate.get("redrob_signals", {})
    oar = signals.get("offer_acceptance_rate", -1)

    if oar == -1:
        return False  # sentinel value, no data

    interview_rate = signals.get("interview_completion_rate", 0)
    signup_date_str = signals.get("signup_date", "")

    # If very high offer acceptance rate but very low interview completion
    # and very new signup, suspicious
    if oar > 0.8 and interview_rate < 0.2:
        try:
            signup = datetime.strptime(signup_date_str, "%Y-%m-%d")
            days_since = (REFERENCE_DATE - signup).days
            if days_since < 30:
                return True
        except (ValueError, TypeError):
            pass

    return False


def is_honeypot(
    candidate: dict,
    company_earliest_starts: dict[str, str] | None = None,
) -> bool:
    """Master honeypot check — returns True if ANY standalone rule fires."""
    standalone_checks = [
        expert_with_zero_duration(candidate),
        duration_exceeds_elapsed_time(candidate),
        total_months_exceeds_yoe(candidate),
        implausible_skill_count(candidate),
        description_title_mismatch(candidate),
        proficiency_assessment_mismatch(candidate),
        endorsements_disproportionate(candidate),
        company_tenure_predates_market_entry(candidate, company_earliest_starts),
        salary_inconsistent_with_seniority(candidate),
        offer_acceptance_without_offers(candidate),
    ]
    if any(standalone_checks):
        return True
            
    return False


def get_honeypot_flags(
    candidate: dict,
    company_earliest_starts: dict[str, str] | None = None,
) -> list[str]:
    """Return list of which honeypot rules fired (for debugging)."""
    rules = [
        ("expert_zero_dur", expert_with_zero_duration),
        ("duration_exceeds_elapsed", duration_exceeds_elapsed_time),
        ("total_months_exceeds_yoe", total_months_exceeds_yoe),
        ("implausible_skill_count", implausible_skill_count),
        ("desc_title_mismatch", description_title_mismatch),
        ("proficiency_assessment_mismatch", proficiency_assessment_mismatch),
        ("endorsements_disproportionate", endorsements_disproportionate),
        ("salary_inconsistent", salary_inconsistent_with_seniority),
        ("offer_without_history", offer_acceptance_without_offers),
    ]
    flags = []
    for name, fn in rules:
        if fn(candidate):
            flags.append(name)

    # Rule 10 needs extra arg
    if company_tenure_predates_market_entry(candidate, company_earliest_starts):
        flags.append("company_tenure_predates_market")

    return flags
