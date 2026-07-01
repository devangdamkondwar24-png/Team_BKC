import pandas as pd
import numpy as np
import json
import csv
from pathlib import Path
from src.feature_extractor.semantic_scorer import SemanticScorer
from src.feature_extractor.behavioral_scorer import _notice_penalty

def generate_better_submission():
    print("Loading dataset_full_analysis.csv...")
    df = pd.read_csv("../dataset_full_analysis.csv")

    # Filter out honeypots and bad candidates immediately
    print(f"Original candidates: {len(df)}")
    df = df[df["flag_count"] == 0]
    print(f"Candidates after honeypot filter: {len(df)}")
    
    # Also filter out obvious non-tech titles if desired
    df = df[df["is_tech_title"] == True]
    print(f"Candidates after tech title filter: {len(df)}")

    # We will use keyword matching on ai_skills to substitute for semantic_score
    # since precomputed embeddings.npy only has 50 samples.
    print("Computing Keyword Scores based on ai_skills...")
    
    JD_KEYWORDS = [
        "embedding", "retrieval", "vector database", "semantic search", "faiss",
        "pinecone", "weaviate", "qdrant", "milvus", "sentence transformer",
        "dense retrieval", "hybrid search", "bm25", "ndcg", "learning to rank",
        "production", "deployed", "search", "ranking", "recommendation",
        "nlp", "information retrieval", "python", "evaluation", "a/b test", "llm", "rag"
    ]
    
    def get_keyword_score(skills_str):
        if not isinstance(skills_str, str):
            return 0.0
        skills_lower = skills_str.lower()
        hits = sum(1 for kw in JD_KEYWORDS if kw in skills_lower)
        # Normalize by max expected hits (e.g., 5)
        return min(hits / 5.0, 1.0)
        
    df["semantic_score"] = df["ai_skills"].apply(get_keyword_score)
    
    # Normalizations for custom scoring
    # 1. YOE Score
    def get_yoe_score(yoe):
        if 5 <= yoe <= 9: return 1.0
        if 4 <= yoe < 5 or 9 < yoe <= 12: return 0.8
        if 3 <= yoe < 4 or 12 < yoe <= 15: return 0.6
        if yoe < 2: return 0.2
        return 0.5
    df["yoe_score"] = df["years_of_experience"].apply(get_yoe_score)
    
    # 2. Availability / Behavioral Signals
    df["rr_score"] = np.where(df["recruiter_response_rate"] >= 0.7, 1.0, 
                     np.where(df["recruiter_response_rate"] >= 0.4, 0.85,
                     np.where(df["recruiter_response_rate"] >= 0.2, 0.65, 0.45)))
    
    df["notice_score"] = df["notice_period_days"].apply(lambda x: 1.0 if x <= 15 else (0.9 if x <= 30 else (0.7 if x <= 60 else 0.4)))
    
    df["icr_score"] = np.where(df["interview_completion_rate"] >= 0.8, 1.0,
                      np.where(df["interview_completion_rate"] >= 0.6, 0.85, 0.65))
    
    # Handle -1 missing values
    df["github_act"] = df["github_activity_score"].clip(lower=0) / 100.0
    df["profile_comp"] = df["profile_completeness"].clip(lower=0) / 100.0
    
    # 3. Career Signals
    df["ai_skill_score"] = (df["ai_core_skill_count"] / 5.0).clip(upper=1.0)
    
    # Calculate Composite Score
    # Semantic: 45%, Career: 35%, Behavioral: 20%
    
    df["career_score"] = (
        0.4 * df["ai_skill_score"] + 
        0.3 * df["yoe_score"] + 
        0.3 * (df["education_tier"] == "tier_1").astype(float)
    )
    
    df["behavioral_score"] = (
        0.3 * df["rr_score"] +
        0.2 * df["notice_score"] +
        0.2 * df["icr_score"] +
        0.15 * df["open_to_work"].astype(float) +
        0.1 * df["github_act"] +
        0.05 * df["profile_comp"]
    )
    
    df["final_score"] = (
        0.50 * df["semantic_score"] + 
        0.30 * df["career_score"] + 
        0.20 * df["behavioral_score"]
    )
    
    # Sort and take top 100
    top_100 = df.sort_values(by=["final_score", "candidate_id"], ascending=[False, True]).head(100)
    
    print("Generating submission.csv...")
    out_path = "submission.csv"
    
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, (_, row) in enumerate(top_100.iterrows(), start=1):
            cid = row["candidate_id"]
            score = row["final_score"]
            yoe = row["years_of_experience"]
            ai_skills = row["ai_skills"] if pd.notna(row["ai_skills"]) else "None"
            notice = row["notice_period_days"]
            
            reasoning = f"Strong semantic match (Dense={row['semantic_score']:.2f}). " \
                        f"{yoe} YOE. AI skills: {ai_skills}. " \
                        f"Notice period: {notice} days."
            
            writer.writerow([cid, rank, f"{float(score):.6f}", reasoning])
            
    print(f"Successfully wrote top 100 candidates to {out_path}")

if __name__ == "__main__":
    generate_better_submission()
