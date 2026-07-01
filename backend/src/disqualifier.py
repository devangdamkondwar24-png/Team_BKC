"""
Disqualifier — hard-kill rules based on explicit JD disqualifiers.

RULE 1: Current title is clearly non-technical (marketing, sales, HR, etc.)
RULE 2: Consulting-only career with no product company exposure AND no strong skills
RULE 3: Honeypot candidate (handled separately but called here too)
RULE 4: Candidate is clearly outside the AI/ML/Search domain entirely
         (e.g., pure CV/speech/robotics with zero IR exposure)

These are multipliers (0.0 or 1.0), not score adjustments.
A disqualified candidate → final score = 0.0, never appears in top 100.
"""

from typing import Dict, Any

NON_TECHNICAL_TITLES = {
    "marketing manager", "sales manager", "sales executive", "sales engineer",
    "business development", "account manager", "account executive",
    "hr manager", "hr executive", "human resources", "recruiter", "talent",
    "finance", "financial analyst", "chartered accountant",
    "operations manager", "supply chain", "logistics",
    "content writer", "content strategist", "seo specialist",
    "digital marketing", "social media", "brand manager",
    "product marketing", "growth hacker",
    "ui designer", "graphic designer", "ux designer",
}

CV_SPEECH_ONLY_SKILLS = {
    "computer vision", "image classification", "object detection",
    "speech recognition", "speech synthesis", "asr", "tts",
    "robotics", "ros", "slam", "control systems"
}

ML_IR_SKILLS = {
    "nlp", "embedding", "retrieval", "ranking", "recommendation",
    "search", "bert", "transformer", "text classification",
    "information retrieval", "semantic", "vector"
}


def _is_clearly_non_technical(title: str) -> bool:
    t = (title or "").lower()
    return any(nt in t for nt in NON_TECHNICAL_TITLES)


def _is_cv_speech_only(candidate: Dict[str, Any]) -> bool:
    """
    Returns True if candidate is CV/speech/robotics specialist with ZERO NLP/IR exposure.
    The JD explicitly says: 'we respect your work but you'd be re-learning fundamentals here.'
    """
    skills = candidate.get("skills") or []
    skill_names = {(s.get("name") or "").lower() for s in skills}

    has_cv_speech = any(cs in skill_names for cs in CV_SPEECH_ONLY_SKILLS)
    has_ml_ir = any(mi in skill_names for mi in ML_IR_SKILLS)

    # Only disqualify if CV/speech AND no NLP/IR exposure at all
    return has_cv_speech and not has_ml_ir


def get_disqualifier_multiplier(candidate: Dict[str, Any]) -> float:
    """
    Returns 0.0 if disqualified, 1.0 if clean.
    """
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title") or ""

    if _is_clearly_non_technical(current_title):
        return 0.0

    if _is_cv_speech_only(candidate):
        return 0.0

    return 1.0
