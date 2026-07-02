"""Profile the candidate dataset to understand data distributions and detect honeypot patterns."""
import json
import sys
from collections import Counter
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = r'c:\Users\darsh\Downloads\Hack_to_Skills\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'

honeypot_flags = []
total = 0

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        c = json.loads(line)
        total += 1
        flags = []
        
        # Check 1: expert proficiency with 0 duration_months
        for s in c.get('skills', []):
            if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0:
                flags.append(f"expert_zero_dur: {s['name']}")
        
        # Check 2: career history duration vs years_of_experience mismatch
        total_career_months = sum(h.get('duration_months', 0) for h in c.get('career_history', []))
        yoe = c['profile']['years_of_experience']
        if total_career_months > (yoe + 3) * 12:
            flags.append(f"career_exceeds_yoe: {total_career_months}mo vs {yoe:.1f}yr ({yoe*12:.0f}mo)")
        
        # Check 3: start_date impossibly early for a company
        for h in c.get('career_history', []):
            start = h.get('start_date', '')
            dur = h.get('duration_months', 0)
            if start and dur:
                try:
                    start_dt = datetime.strptime(start, '%Y-%m-%d')
                    # If claim is e.g. started at a company in 2000 but company was founded much later
                    # We can check if duration_months exceeds the time from start_date to now
                    now = datetime(2026, 6, 1)
                    max_months = (now.year - start_dt.year) * 12 + (now.month - start_dt.month)
                    if dur > max_months + 3:
                        flags.append(f"duration_exceeds_elapsed: {dur}mo at {h['company']} (start {start}, max possible ~{max_months}mo)")
                except:
                    pass
        
        # Check 4: Title-description mismatch (non-engineering title with engineering description)
        # This is more subtle - skip for profiling, handle in scoring
        
        if flags:
            honeypot_flags.append((c['candidate_id'], c['profile']['current_title'], flags))

print(f"Total candidates: {total}")
print(f"Total flagged candidates: {len(honeypot_flags)}")
print()

# Show first 20 flagged
print("=== First 20 flagged candidates ===")
for cid, title, flags in honeypot_flags[:20]:
    print(f"{cid}: {title}")
    for f in flags:
        print(f"  FLAG: {f}")
    print()

# Count flag types
flag_type_counts = Counter()
for _, _, flags in honeypot_flags:
    for f in flags:
        flag_type = f.split(':')[0]
        flag_type_counts[flag_type] += 1

print("=== Flag type distribution ===")
for ft, count in flag_type_counts.most_common():
    print(f"  {ft}: {count}")
