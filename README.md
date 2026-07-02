# Redrob Intelligent Candidate Discovery & Ranking System

This repository contains a CPU-only candidate ranking system designed to process 100,000 candidate profiles in under 5 minutes, adhering strictly to the constraints of the Redrob AI challenge.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Offline Precomputation (One-Time)
Because semantic embedding of 100,000 profiles is too slow for the 5-minute budget, we precompute the text embeddings and a few expensive aggregates offline.
```bash
python precompute.py --candidates ./candidates.jsonl
```
*Note: This will download the `all-MiniLM-L6-v2` model and process the entire dataset. It takes ~1.5 hours on CPU. It produces a `precomputed/` directory with `.npy` matrices.*

### 3. Ranking Generation (< 5 minutes)
Once precomputation is done, run the main ranking script to generate the final submission CSV.
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
This applies the fast-prefilter (eliminating non-tech profiles instantly), loads the precomputed embeddings for the remaining candidates, computes cosine similarities, applies the structured/career scoring rules, gates out honeypots, and writes the top 100 to the output CSV with generated reasoning strings.

### 4. Validation
Verify the final submission format:
```bash
python validate_submission.py submission.csv
```

## Architecture

The system uses a **five-pillar** scoring approach, meticulously designed around the explicit requirements in the Job Description (JD).

1. **Career Depth (40%)**: The single strongest differentiator. Scans career history descriptions for keywords indicating production deployment, scales, and live users. Heavily penalizes candidates whose entire career is at IT services firms. Rewards experience specifically in the retrieval/ranking domain.
2. **Semantic Similarity (25%)**: Uses `all-MiniLM-L6-v2` (384-dim) to compute cosine similarity between the JD text and the candidate's profile/career descriptions. Captures candidates who describe relevant work without using buzzwords.
3. **Structured Fit (25%)**: Extracted feature scoring covering years of experience (5-9 yr sweet spot), must-have skills (retrieval, vector DBs, Python, eval frameworks), nice-to-have skills, education tier, and location matching.
4. **Behavioral Signals (10%)**: A composite of Redrob platform signals representing candidate reachability and availability (recency, open-to-work flag, recruiter response rate).
5. **Disqualifier Gate**: A multiplicative penalty applied at the end for JD-stated dealbreakers (e.g., pure research, LangChain-only, senior without code, non-tech career).

**Honeypot Detector**: A strict, independent gate containing 12 deterministic rules. If a candidate triggers *any* rule (e.g., expert proficiency with 0 duration, or salary ranges vastly inverted), they are hard-zeroed and excluded.

## Design Tradeoffs
- **Model Choice**: We opted for `all-MiniLM-L6-v2` over larger models (like `mpnet-base-v2`) to keep precomputation wall-clock times reasonable on CPU while preserving enough semantic separation.
- **Top-K Cutoff**: We deliberately removed any early semantic similarity cutoff (e.g., top 2000). The fast pre-filter drops obvious non-matches (accountants, HR), but we run the full regex and rule-based structured scorer on *all* remaining candidates. This prevents losing "plain-language" fits who might have low cosine similarity but strong production experience.
- **Dynamic Reasoning**: Instead of boilerplate templates, reasoning strings are generated dynamically by pulling the highest-scoring signals directly from the candidate's JSON (e.g., specific matched must-have skills and career highlights).
