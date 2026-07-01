from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import csv
import json
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/candidates")
def get_candidates():
    # Load candidates from JSON
    candidates_dict = {}
    json_path = Path(r"c:\Users\darsh\Downloads\Hack_to_Skill\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\sample_candidates.json")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            candidates_data = json.load(f)
            for c in candidates_data:
                candidates_dict[c['candidate_id']] = c
    except FileNotFoundError:
        pass # Handle gracefully if file isn't there
            
    # Load rankings from submission.csv
    results = []
    csv_path = Path("submission.csv")
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row['candidate_id']
                c_data = candidates_dict.get(cid, {})
                profile = c_data.get('profile', {})
                signals = c_data.get('redrob_signals', {})
                skills = [s['name'] for s in c_data.get('skills', [])][:5]
                
                results.append({
                    "candidate_id": cid,
                    "rank": int(row['rank']),
                    "score": float(row['score']),
                    "reasoning": row['reasoning'],
                    "title": profile.get("current_title", "Unknown Role"),
                    "company": profile.get("current_company", "Unknown Company"),
                    "open_to_work": signals.get("open_to_work_flag", False),
                    "notice_period": signals.get("notice_period_days", 30),
                    "skills": skills
                })
    except FileNotFoundError:
        return {"error": "submission.csv not found. Please run the ranking pipeline first."}
            
    # Sort by rank
    results.sort(key=lambda x: x['rank'])
    return {"candidates": results}
