"""
Disqualifier — Multiplicative penalty for JD-stated dealbreakers.

Applied AFTER the additive composite, BEFORE honeypot zeroing.
Multiple disqualifiers compound (product of factors), floored at 0.05.
"""
from src.config import (
    IT_SERVICES_COMPANIES, NON_TECH_TITLES, AI_ENGINEERING_TITLES,
    SOFTWARE_ENGINEERING_TITLES, OFFDOMAIN_SPECIALIST_SKILLS, NLP_IR_SKILLS,
    PRODUCTION_KEYWORDS, CORE_AI_SKILLS,
)


def _is_pure_research_no_prod(candidate: dict) -> bool:
    """Career in academic/research-only roles with no production deployment signals."""
    career = candidate.get("career_history", [])
    if not career:
        return False
    research_kw = {"research", "academic", "lab", "phd", "postdoc", "professor",
                   "publication", "paper", "thesis", "journal"}
    has_research = False
    has_production = False
    for job in career:
        desc = job.get("description", "").lower()
        title = job.get("title", "").lower()
        if any(kw in desc or kw in title for kw in research_kw):
            has_research = True
        if any(kw in desc for kw in PRODUCTION_KEYWORDS):
            has_production = True
    return has_research and not has_production


def _is_langchain_wrapper_only(candidate: dict) -> bool:
    """AI experience <12 months AND primarily LangChain/OpenAI wrapper work."""
    cand_skills = {s.get("name", "").lower() for s in candidate.get("skills", [])}
    ai_skills = {s.lower() for s in CORE_AI_SKILLS}
    wrapper_skills = {"langchain", "openai", "chatgpt", "gpt-4", "llamaindex"}

    has_wrapper = bool(cand_skills & wrapper_skills)
    has_deep_ai = bool(cand_skills & (ai_skills - wrapper_skills))

    if not has_wrapper:
        return False
    if has_deep_ai:
        return False

    # Check career descriptions for shallow AI work
    total_ai_months = 0
    for job in candidate.get("career_history", []):
        desc = job.get("description", "").lower()
        if any(w in desc for w in ("langchain", "openai", "gpt", "llm", "chatbot")):
            total_ai_months += job.get("duration_months", 0)
    return total_ai_months < 12


def _is_senior_architect_no_code(candidate: dict) -> bool:
    """Senior title but career descriptions suggest no coding in 18+ months."""
    title = candidate.get("profile", {}).get("current_title", "").lower()
    if "architect" not in title and "director" not in title and "vp" not in title:
        return False

    career = candidate.get("career_history", [])
    if not career:
        return False

    # Check most recent role
    current = career[0] if career else {}
    desc = current.get("description", "").lower()
    code_kw = {"code", "coding", "implement", "built", "developed", "wrote",
               "python", "java", "typescript", "rust", "go ", "c++", "programming"}
    management_kw = {"managed", "led team", "oversaw", "strategy", "roadmap",
                     "stakeholder", "budget", "headcount"}

    has_code = any(kw in desc for kw in code_kw)
    has_mgmt = any(kw in desc for kw in management_kw)

    return has_mgmt and not has_code


def _is_all_it_services(candidate: dict) -> bool:
    """Entire career at IT services firms with no product-company experience."""
    career = candidate.get("career_history", [])
    if not career:
        return False
    total = 0
    services = 0
    for job in career:
        dur = job.get("duration_months", 0)
        total += dur
        company = job.get("company", "")
        industry = job.get("industry", "")
        if company in IT_SERVICES_COMPANIES or industry == "IT Services":
            services += dur
    if total == 0:
        return False
    return (services / total) >= 0.95


def _is_offdomain_specialist(candidate: dict) -> bool:
    """CV/Speech/Robotics specialist with no NLP/IR exposure."""
    cand_skills = {s.get("name", "") for s in candidate.get("skills", [])}
    offdomain = cand_skills & OFFDOMAIN_SPECIALIST_SKILLS
    nlp_ir = cand_skills & NLP_IR_SKILLS
    if len(offdomain) >= 3 and len(nlp_ir) == 0:
        return True
    return False


def _is_closed_source_unvalidated(candidate: dict) -> bool:
    """5+ years on closed-source with zero external validation."""
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if yoe < 5:
        return False
    signals = candidate.get("redrob_signals", {})
    github = signals.get("github_activity_score", -1)
    certs = candidate.get("certifications", [])
    # No GitHub, no certifications → closed-source unvalidated
    if github <= 0 and len(certs) == 0:
        # Check career descriptions for external validation signals
        all_desc = " ".join(
            j.get("description", "") for j in candidate.get("career_history", [])
        ).lower()
        validation_kw = {"open source", "paper", "publication", "conference",
                         "talk", "blog", "github", "contributed"}
        if not any(kw in all_desc for kw in validation_kw):
            return True
    return False


def _is_non_tech(candidate: dict) -> bool:
    """Non-technical title AND non-technical career history."""
    title = candidate.get("profile", {}).get("current_title", "")
    if title not in NON_TECH_TITLES:
        return False
    # Check if any career role is technical
    for job in candidate.get("career_history", []):
        jt = job.get("title", "")
        if jt in AI_ENGINEERING_TITLES or jt in SOFTWARE_ENGINEERING_TITLES:
            return False  # Has some tech background
    return True


# Rule table: (name, detector, multiplier when triggered)
DISQUALIFIER_RULES = [
    ("pure_research_no_production", _is_pure_research_no_prod, 0.10),
    ("langchain_wrapper_only", _is_langchain_wrapper_only, 0.20),
    ("senior_no_code_18mo", _is_senior_architect_no_code, 0.15),
    ("all_it_services_no_product", _is_all_it_services, 0.20),
    ("cv_speech_robotics_no_nlp", _is_offdomain_specialist, 0.30),
    ("closed_source_no_validation", _is_closed_source_unvalidated, 0.50),
    ("non_tech_title_and_career", _is_non_tech, 0.05),
]


def disqualifier_multiplier(candidate: dict) -> float:
    """Compute multiplicative penalty (0.05-1.0). Multiple rules compound."""
    multiplier = 1.0
    for name, detector, factor in DISQUALIFIER_RULES:
        if detector(candidate):
            multiplier *= factor
    return max(multiplier, 0.05)


def get_disqualifier_flags(candidate: dict) -> list[str]:
    """Return list of which disqualifier rules fired (for reasoning)."""
    return [name for name, det, _ in DISQUALIFIER_RULES if det(candidate)]
