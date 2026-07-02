"""
Honeypot Calibration Script

Runs the honeypot detector against the sample_candidates.json
to check detection rates and false positives.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.honeypot_detector import is_honeypot, get_honeypot_flags

def main():
    sample_path = r".\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\sample_candidates.json"
    
    try:
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_candidates = json.load(f)
    except Exception as e:
        print(f"Error loading {sample_path}: {e}")
        return

    print(f"Loaded {len(sample_candidates)} sample candidates.")
    
    flagged_candidates = []
    rule_counts = {}
    
    for i, cand in enumerate(sample_candidates):
        flags = get_honeypot_flags(cand, None)
        if flags:
            flagged_candidates.append((i, cand, flags))
            for f in flags:
                rule_counts[f] = rule_counts.get(f, 0) + 1
            
    print(f"\nFlagged {len(flagged_candidates)} out of {len(sample_candidates)} candidates.")
    print("\n--- Rule Breakdown ---")
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"{rule}: {count}")

    print("\n--- Spot Check (First 10) ---")
    for idx, cand, flags in flagged_candidates[:10]:
        cid = cand["candidate_id"]
        title = cand.get("profile", {}).get("current_title", "")
        yoe = cand.get("profile", {}).get("years_of_experience", 0)
        salary = cand.get("redrob_signals", {}).get("expected_salary_range_inr_lpa", {})
        edu = [(e.get("degree"), e.get("start_year"), e.get("end_year")) for e in cand.get("education", [])]
        careers = [(j.get("start_date"), j.get("duration_months")) for j in cand.get("career_history", [])]
        print(f"\n[{cid}] Flags: {flags}")
        print(f"Title: {title}, YoE: {yoe}")
        if "salary_inverted_or_absurd" in flags:
            print(f"Salary: {salary}")
        if "education_overlaps_career" in flags:
            print(f"Edu: {edu}")
            print(f"Careers: {careers}")
        if "proficiency_assessment_mismatch" in flags:
            print("Proficiencies vs Assessments mismatch triggered")

if __name__ == "__main__":
    main()
