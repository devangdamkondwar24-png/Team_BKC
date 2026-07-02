"""Profile AI-relevant candidates to understand what strong fits look like."""
import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = r'c:\Users\darsh\Downloads\Hack_to_Skills\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl'

AI_TITLES = {'ML Engineer', 'AI Research Engineer', 'Data Scientist', 'Senior Software Engineer (ML)',
             'Junior ML Engineer', 'AI Specialist', 'Machine Learning Engineer', 'Applied ML Engineer',
             'AI Engineer', 'Senior Data Scientist', 'NLP Engineer', 'Senior NLP Engineer',
             'Senior Machine Learning Engineer', 'Staff Machine Learning Engineer', 'Senior AI Engineer',
             'Lead AI Engineer', 'Search Engineer', 'Ranking Engineer', 'MLOps Engineer'}

RETRIEVAL_SKILLS = {'FAISS', 'Pinecone', 'Weaviate', 'Qdrant', 'Milvus', 'OpenSearch', 'Elasticsearch',
                    'sentence-transformers', 'Vector DB', 'Embedding', 'BGE', 'E5', 'RAG',
                    'Information Retrieval', 'Semantic Search', 'LangChain'}

CORE_ML_SKILLS = {'PyTorch', 'TensorFlow', 'Scikit-learn', 'XGBoost', 'LightGBM', 'Hugging Face',
                  'NLP', 'Computer Vision', 'Deep Learning', 'Machine Learning', 'Neural Networks',
                  'Transformers', 'BERT', 'GPT', 'Fine-tuning LLMs', 'LoRA', 'MLflow', 'Weights & Biases',
                  'Feature Engineering', 'Statistical Modeling', 'Recommendation Systems'}

PYTHON_SKILLS = {'Python', 'FastAPI', 'Flask', 'Django'}

ai_candidates = []

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        c = json.loads(line)
        title = c['profile']['current_title']
        skills = {s['name'] for s in c.get('skills', [])}
        
        has_ai_title = title in AI_TITLES
        retrieval_overlap = skills & RETRIEVAL_SKILLS
        ml_overlap = skills & CORE_ML_SKILLS
        python_overlap = skills & PYTHON_SKILLS
        
        if has_ai_title or len(retrieval_overlap) >= 1 or len(ml_overlap) >= 3:
            ai_candidates.append({
                'id': c['candidate_id'],
                'title': title,
                'company': c['profile']['current_company'],
                'industry': c['profile']['current_industry'],
                'yoe': c['profile']['years_of_experience'],
                'location': c['profile']['location'],
                'country': c['profile']['country'],
                'skills': skills,
                'retrieval_skills': retrieval_overlap,
                'ml_skills': ml_overlap,
                'python': python_overlap,
                'has_ai_title': has_ai_title,
            })

print(f"AI-relevant candidates: {len(ai_candidates)}")
print()

# Show some strong fits
print("=== Candidates with AI titles AND retrieval skills ===")
strong = [c for c in ai_candidates if c['has_ai_title'] and c['retrieval_skills']]
print(f"Count: {len(strong)}")
for c in strong[:10]:
    print(f"  {c['id']}: {c['title']} at {c['company']} ({c['industry']})")
    print(f"    Location: {c['location']}, {c['country']} | YoE: {c['yoe']}")
    print(f"    Retrieval: {c['retrieval_skills']}")
    print(f"    ML: {c['ml_skills']}")
    print(f"    Python: {c['python']}")
    print()

# Title distribution of AI candidates
print("=== Title distribution of AI-relevant pool ===")
title_dist = Counter(c['title'] for c in ai_candidates)
for t, cnt in title_dist.most_common(20):
    print(f"  {t}: {cnt}")

# Industry distribution
print("\n=== Industry distribution of AI-relevant pool ===")
ind_dist = Counter(c['industry'] for c in ai_candidates)
for i, cnt in ind_dist.most_common(10):
    print(f"  {i}: {cnt}")

# Country distribution
print("\n=== Country/Location of AI candidates ===")
country_dist = Counter(c['country'] for c in ai_candidates)
for co, cnt in country_dist.most_common(10):
    print(f"  {co}: {cnt}")
