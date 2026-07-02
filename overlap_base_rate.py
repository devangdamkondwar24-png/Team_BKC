import json
import numpy as np
from datetime import datetime

def check_overlap(candidate):
    overlap_months_list = []
    for edu in candidate.get("education", []):
        end_year = edu.get("end_year", 0)
        start_year = edu.get("start_year", 0)
        if not end_year or not start_year:
            continue
        degree = edu.get("degree", "").lower()
        if degree in ("ph.d", "phd", "m.tech", "m.sc", "m.e.", "mba", "b.tech", "b.e.", "b.sc"):
            for job in candidate.get("career_history", []):
                job_start = job.get("start_date", "")
                if not job_start:
                    continue
                title = job.get("title", "").lower()
                if "intern" in title or "trainee" in title:
                    continue
                try:
                    job_start_dt = datetime.strptime(job_start, "%Y-%m-%d")
                    if job_start_dt.year < end_year:
                        degree_end = datetime(end_year, 6, 30)
                        if job_start_dt < degree_end:
                            overlap_days = (degree_end - job_start_dt).days
                            overlap_months = min(job.get("duration_months", 0), overlap_days / 30.0)
                            if overlap_months > 0:
                                overlap_months_list.append(overlap_months)
                except:
                    continue
    return sum(overlap_months_list) if overlap_months_list else 0

def main():
    path = r".\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    overlaps = []
    total = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            total += 1
            cand = json.loads(line)
            ov = check_overlap(cand)
            if ov > 0:
                overlaps.append(ov)
                
    has_overlap_pct = (len(overlaps) / total) * 100
    print(f"Total candidates: {total}")
    print(f"Candidates with >0 overlap: {len(overlaps)} ({has_overlap_pct:.2f}%)")
    if overlaps:
        print(f"Min: {np.min(overlaps):.2f}")
        print(f"Median: {np.median(overlaps):.2f}")
        print(f"P90: {np.percentile(overlaps, 90):.2f}")
        print(f"P99: {np.percentile(overlaps, 99):.2f}")
        print(f"Max: {np.max(overlaps):.2f}")

if __name__ == "__main__":
    main()
