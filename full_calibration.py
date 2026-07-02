import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.honeypot_detector import is_honeypot, get_honeypot_flags

def main():
    cands_path = r".\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    
    print("Loading all candidates (this may take a few seconds)...")
    all_cands = []
    with open(cands_path, 'r', encoding='utf-8') as f:
        for line in f:
            all_cands.append(json.loads(line))
            
    print(f"Total candidates loaded: {len(all_cands)}")
    
    # Let's build the company_earliest_starts table for Step 2!
    company_earliest_starts = {}
    for cand in all_cands:
        for job in cand.get("career_history", []):
            company = job.get("company", "").strip().lower()
            start_date = job.get("start_date", "")
            if not company or not start_date:
                continue
            if company not in company_earliest_starts:
                company_earliest_starts[company] = start_date
            else:
                if start_date < company_earliest_starts[company]:
                    company_earliest_starts[company] = start_date
                    
    print(f"Built company_earliest_starts table with {len(company_earliest_starts)} companies.")
    
    sample_size = len(all_cands)
    random.seed(42)
    sample_cands = all_cands
    print(f"\nProcessing all {sample_size} candidates.")
    
    flagged_candidates = []
    rule_counts = {}
    
    for i, cand in enumerate(sample_cands):
        flags = get_honeypot_flags(cand, company_earliest_starts)
        if flags:
            flagged_candidates.append((i, cand, flags))
            for f in flags:
                rule_counts[f] = rule_counts.get(f, 0) + 1
            
    print(f"\nFlagged {len(flagged_candidates)} out of {sample_size} candidates.")
    flag_rate = (len(flagged_candidates) / sample_size) * 100
    print(f"Flag Rate: {flag_rate:.4f}%")
    expected = (sample_size / 100000.0) * 80
    print(f"Expected count for comparison: {expected:.2f}")
    
    print("\n--- Rule Breakdown ---")
    for rule, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        print(f"{rule}: {count}")

    print("\n--- Testing Specific Rules on Synthetic Data (Step 3) ---")
    # Rule 1: expert_zero_dur
    c1 = {"skills": [{"name": "Python", "proficiency": "expert", "last_used": 2024}], "career_history": []}
    flags1 = get_honeypot_flags(c1, company_earliest_starts)
    print(f"expert_zero_dur fires: {'expert_zero_dur' in flags1}")
    
    # Rule 2: duration_exceeds_elapsed
    c2 = {"career_history": [{"start_date": "2023-01-01", "end_date": "2024-01-01", "duration_months": 36}]}
    flags2 = get_honeypot_flags(c2, company_earliest_starts)
    print(f"duration_exceeds_elapsed fires: {'duration_exceeds_elapsed' in flags2}")
    
    # Rule 3: total_months_exceeds_yoe
    c3 = {"profile": {"years_of_experience": 2.0}, "career_history": [{"start_date": "2020-01-01", "duration_months": 60}]}
    flags3 = get_honeypot_flags(c3, company_earliest_starts)
    print(f"total_months_exceeds_yoe fires: {'total_months_exceeds_yoe' in flags3}")
    
    # Rule 4: implausible_skill_count
    c4 = {"skills": [{"name": str(i), "proficiency": "expert"} for i in range(25)]}
    flags4 = get_honeypot_flags(c4, company_earliest_starts)
    print(f"implausible_skill_count fires: {'implausible_skill_count' in flags4}")
    
    # Rule 5: desc_title_mismatch
    c5 = {"career_history": [{"title": "Software Engineer", "description": "gaap ind-as general ledger month-end close tax filings"}]}
    flags5 = get_honeypot_flags(c5, company_earliest_starts)
    print(f"desc_title_mismatch fires: {'desc_title_mismatch' in flags5}")
    
    # Rule 6: education_chronology_impossible
    c6 = {"education": [{"degree": "B.Tech", "start_year": 2022, "end_year": 2020}, {"degree": "Ph.D", "start_year": 2010, "end_year": 2014}]}
    flags6 = get_honeypot_flags(c6, company_earliest_starts)
    print(f"education_chronology_impossible fires: {'education_chronology_impossible' in flags6}")
    
    # Rule 8: education_overlap_extreme
    c6b = {"education": [{"degree": "B.Tech", "start_year": 2018, "end_year": 2022}], "career_history": [{"start_date": "2018-08-01", "duration_months": 48}]}
    flags6b = get_honeypot_flags(c6b, company_earliest_starts)
    print(f"education_overlap_extreme fires: {'education_overlap_extreme' in flags6b}")
    
    # Rule: endorsements_disproportionate
    c7 = {"redrob_signals": {"profile_completeness_score": 10, "peer_endorsement_count": 250, "endorsements_received": 250}, "profile": {"years_of_experience": 1.0}}
    flags7 = get_honeypot_flags(c7, company_earliest_starts)
    print(f"endorsements_disproportionate fires: {'endorsements_disproportionate' in flags7}")
    
    # Rule: offer_without_history
    c8 = {"redrob_signals": {"offer_acceptance_rate": 0.9, "interview_completion_rate": 0.1, "signup_date": "2026-06-25"}}
    flags8 = get_honeypot_flags(c8, company_earliest_starts)
    print(f"offer_without_history fires: {'offer_without_history' in flags8}")

    print("\n--- Spot Check for company_tenure_predates_market_entry ---")
    rule10_flags = [cand for cand in flagged_candidates if "company_tenure_predates_market_entry" in cand[2]]
    for idx, cand, flags in rule10_flags[:5]:
        print(f"[{cand['candidate_id']}]")
        for job in cand.get("career_history", []):
            comp = job.get("company_name", "").strip().lower()
            st = job.get("start_date", "")
            earliest = company_earliest_starts.get(comp, "N/A")
            if comp and st and st < earliest[:4]: # Wait, earliest is full date, we check year < earliest year - 3
                earliest_year = int(earliest[:4])
                start_year = int(st[:4])
                if start_year < earliest_year - 3:
                    print(f"  Job at '{comp}': start {st}, but dataset earliest is {earliest} (Predates by {earliest_year - start_year} years)")

if __name__ == "__main__":
    main()
