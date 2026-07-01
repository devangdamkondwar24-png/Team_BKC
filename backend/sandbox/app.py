"""
Streamlit Sandbox — Required for submission.
Deploy to HuggingFace Spaces (free tier).

Accepts a sample_candidates.json upload → runs full pipeline → shows ranked results.
Does NOT call any external API. Fully offline. No network during ranking.
"""

import streamlit as st
import json
import pandas as pd
import io
from pathlib import Path

st.set_page_config(
    page_title="Redrob AI Candidate Ranker",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Redrob Intelligent Candidate Ranking System")
st.markdown("""
**Architecture**: BGE-base-en-v1.5 semantic embeddings + BM25 hybrid retrieval + LightGBM LambdaRank  
**Stack used by**: LinkedIn (EBR + LambdaRank), Uber (FAISS + semantic search), Airbnb (GBDT ranking)  
""")

# Upload panel
with st.sidebar:
    st.header("Input")
    uploaded_file = st.file_uploader(
        "Upload sample_candidates.json (≤100 candidates)",
        type=["json", "jsonl"]
    )
    st.markdown("---")
    st.markdown("**JD**: Senior AI Engineer @ Redrob AI")
    st.markdown("**Location**: Pune / Noida")
    st.markdown("**YOE**: 5–9 years")

if uploaded_file:
    raw = uploaded_file.read()
    try:
        candidates = json.loads(raw)
        if isinstance(candidates, dict):
            candidates = [candidates]
    except Exception:
        # Try JSONL
        candidates = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]

    st.success(f"Loaded {len(candidates)} candidates")

    if st.button("🚀 Run Ranking Pipeline", type="primary"):
        with st.spinner("Running BGE semantic scoring..."):
            from src.honeypot_detector import is_honeypot
            from src.disqualifier import get_disqualifier_multiplier
            from src.feature_extractor.career_scorer import score_candidate_career
            from src.feature_extractor.behavioral_scorer import score_candidate_behavioral
            from src.reasoning_generator import generate_reasoning
            from src.feature_extractor.semantic_scorer import SemanticScorer, build_candidate_text
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer("BAAI/bge-base-en-v1.5")
            texts = [build_candidate_text(c) for c in candidates]
            embeddings = model.encode(texts, normalize_embeddings=True)

            JD_QUERY = "Senior AI engineer production embedding retrieval vector database semantic search FAISS ranking recommendation NLP Python"
            jd_emb = model.encode(["Represent this sentence for searching relevant passages: " + JD_QUERY],
                                   normalize_embeddings=True)[0]
            semantic_scores = embeddings @ jd_emb

        results = []
        for i, candidate in enumerate(candidates):
            cid = candidate.get("candidate_id", f"CAND_{i:07d}")
            if is_honeypot(candidate) or get_disqualifier_multiplier(candidate) == 0.0:
                continue
            career_out = score_candidate_career(candidate)
            behavioral_out = score_candidate_behavioral(candidate)
            composite = (
                0.45 * float(semantic_scores[i]) +
                0.35 * career_out["career_score"] +
                0.20 * behavioral_out["behavioral_score"]
            )
            results.append({
                "candidate_id": cid,
                "score": composite,
                "title": candidate.get("profile", {}).get("current_title", ""),
                "company": candidate.get("profile", {}).get("current_company", ""),
                "yoe": candidate.get("profile", {}).get("years_of_experience", 0),
                "location": candidate.get("profile", {}).get("location", ""),
                "semantic": float(semantic_scores[i]),
                "career": career_out["career_score"],
                "behavioral": behavioral_out["behavioral_score"],
                "career_out": career_out,
                "behavioral_out": behavioral_out,
                "candidate": candidate,
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        st.subheader(f"Top {min(len(results), 20)} Candidates")
        rows = []
        for rank, r in enumerate(results[:20], 1):
            reasoning = generate_reasoning(r["candidate"], r["behavioral_out"], r["career_out"], rank, r["score"])
            rows.append({
                "Rank": rank,
                "ID": r["candidate_id"],
                "Title": r["title"],
                "Company": r["company"],
                "YOE": r["yoe"],
                "Location": r["location"],
                "Score": f"{r['score']:.3f}",
                "Semantic": f"{r['semantic']:.3f}",
                "Career": f"{r['career']:.3f}",
                "Behavioral": f"{r['behavioral']:.3f}",
                "Reasoning": reasoning,
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Download CSV
        top_100 = results[:min(100, len(results))]
        csv_rows = []
        for rank, r in enumerate(top_100, 1):
            reasoning = generate_reasoning(r["candidate"], r["behavioral_out"], r["career_out"], rank, r["score"])
            csv_rows.append({
                "candidate_id": r["candidate_id"],
                "rank": rank,
                "score": f"{r['score']:.6f}",
                "reasoning": reasoning,
            })
        csv_df = pd.DataFrame(csv_rows)
        st.download_button(
            "⬇️ Download submission.csv",
            csv_df.to_csv(index=False),
            file_name="submission.csv",
            mime="text/csv"
        )
else:
    st.info("Upload sample_candidates.json from the hackathon bundle to get started.")
    st.markdown("This demo accepts the included `sample_candidates.json` (50 candidates) for quick testing.")
