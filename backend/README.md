# AI Recruiter — Backend

An AI-powered candidate ranking system that evaluates candidates the way a great recruiter would — not by matching keywords, but by understanding who actually fits the role.

## Architecture

```
PDF/JSON Resumes ──► Ingestion Pipeline ──► OpenAI text-embedding-3-large
                                                    │
                                                    ▼
                                            Pinecone Vector Store
                                                    │
                              Job Description ──► Hybrid Retrieval
                                              (65% Semantic + 35% BM25)
                                                    │
                                                    ▼
                                          Top-N Candidate Pool
                                                    │
                                         GPT-4o Judgment Layer
                                       (5 Dimensions, Evidence-based)
                                                    │
                                                    ▼
                                         Final Ranked Results
```

## Setup

### 1. Clone and install dependencies

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=ai-recruiter-index
```

### 3. Run the server

```bash
python main.py
# or: uvicorn main:app --reload
```

API docs available at: http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/candidates/upload` | Upload resumes (PDF/JSON) |
| POST | `/api/rank` | Rank candidates for a job |

### Upload Candidates (POST /api/candidates/upload)

Upload one or more resume files (multipart/form-data):

```bash
curl -X POST http://localhost:8000/api/candidates/upload \
  -F "files=@resume1.pdf" \
  -F "files=@profile2.json"
```

### Rank Candidates (POST /api/rank)

```bash
curl -X POST http://localhost:8000/api/rank \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Senior ML Engineer",
    "description": "We are looking for a senior ML engineer to build our recommendation systems...",
    "hard_requirements": ["5 years Python", "production ML experience"],
    "top_k": 10,
    "run_llm_judge": true,
    "semantic_weight": 0.65,
    "bm25_weight": 0.35
  }'
```

## The 5 LLM Evaluation Dimensions

1. **Technical Depth** — Does this person actually understand the craft, not just name tools?
2. **Learning Velocity** — How fast do they grow in new territory?
3. **Complexity Handled** — What's the hardest problem they've actually solved?
4. **Ownership Pattern** — Do they drive things or just contribute?
5. **Signal-to-Noise Ratio** — Is what they claim backed by what they show?

## File Structure

```
backend/
├── main.py          # FastAPI app — routes and server entry point
├── ingestion.py     # PDF/JSON parsing pipeline
├── vector_store.py  # Pinecone client + OpenAI embedding generation
├── retrieval.py     # Hybrid BM25 + Semantic search engine
├── judge.py         # GPT-4o judgment layer across 5 dimensions
├── models.py        # Pydantic data models
├── config.py        # Environment variable config
├── requirements.txt # Pinned dependencies
└── .env.example     # Environment variable template
```
