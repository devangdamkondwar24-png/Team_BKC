import json
import sys
sys.path.append('.')
from src.honeypot_detector import education_chronology_impossible

def main():
    path = r".\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            cand = json.loads(line)
            if education_chronology_impossible(cand):
                count += 1
                print(f"--- CANDIDATE {count} ---")
                for edu in cand.get("education", []):
                    print(f"  {edu.get('degree')} | {edu.get('start_year')} - {edu.get('end_year')}")
                print()
                if count >= 15:
                    break

if __name__ == "__main__":
    main()
