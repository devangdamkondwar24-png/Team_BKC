"""
Configuration: all weights, thresholds, skill lists, and constants.

Design rationale documented inline for interview defensibility.
"""
from datetime import datetime

# ============================================================================
# Composite Score Weights
# ============================================================================
# NDCG@10 = 50% of eval → career depth (strongest JD differentiator) gets
# highest weight. Semantic similarity captures "reads between the lines"
# candidates. Behavioral is a reachability modifier, not a quality signal.
W_SEMANTIC = 0.25
W_CAREER_DEPTH = 0.40
W_STRUCTURED_FIT = 0.25
W_BEHAVIORAL = 0.10
# sum = 1.00; disqualifiers are multiplicative, not additive

# ============================================================================
# Embedding Model
# ============================================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ============================================================================
# Skill Taxonomy — derived directly from JD
# ============================================================================

# Must-have skill families (JD: "Things you absolutely need")
MUST_HAVE_SKILLS = {
    "embeddings_retrieval": {
        "sentence-transformers", "OpenAI Embeddings", "BGE", "E5",
        "Embedding", "Embeddings", "sentence-transformers",
        "BERT", "Transformers", "Information Retrieval",
        "Semantic Search", "Vector Search",
    },
    "vector_db_hybrid_search": {
        "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
        "Elasticsearch", "FAISS", "Vector DB", "Hybrid Search",
        "Chroma", "pgvector",
    },
    "python": {
        "Python", "FastAPI", "Flask", "Django",
    },
    "eval_frameworks": {
        "NDCG", "MRR", "MAP", "A/B Testing", "Ranking Evaluation",
        "Offline Evaluation", "Online Evaluation", "Evaluation Frameworks",
        "Information Retrieval", "Ranking",
    },
}

# Nice-to-have skills (JD: "Things we'd like you to have")
NICE_TO_HAVE_SKILLS = {
    "llm_finetuning": {
        "LoRA", "QLoRA", "PEFT", "Fine-tuning LLMs", "Fine-tuning",
        "LLM Fine-tuning",
    },
    "learning_to_rank": {
        "XGBoost", "LightGBM", "Learning to Rank", "LTR",
        "Ranking Models", "CatBoost",
    },
    "hrtech_marketplace": {
        "HR Tech", "Recruiting", "Marketplace", "Talent",
        "ATS", "Applicant Tracking",
    },
    "distributed_systems": {
        "Distributed Systems", "Kubernetes", "Docker", "Kafka",
        "Spark", "Ray", "Dask", "Microservices",
    },
    "open_source": {
        "Open Source", "GitHub", "Open-source",
    },
}

# Core ML/AI skills (broader set for detecting AI relevance)
CORE_AI_SKILLS = {
    "PyTorch", "TensorFlow", "Scikit-learn", "Keras",
    "NLP", "Computer Vision", "Deep Learning", "Machine Learning",
    "Neural Networks", "Transformers", "BERT", "GPT",
    "Hugging Face", "MLflow", "Weights & Biases",
    "Feature Engineering", "Statistical Modeling",
    "Recommendation Systems", "RAG", "LangChain",
    "Reinforcement Learning", "Time Series",
    "Natural Language Processing",
}

# Skills that indicate CV/Speech/Robotics specialization (JD: "explicitly do NOT want")
OFFDOMAIN_SPECIALIST_SKILLS = {
    "OpenCV", "Image Classification", "Object Detection",
    "Image Segmentation", "YOLO", "GANs", "Stable Diffusion",
    "Speech Recognition", "TTS", "ASR", "Speech Synthesis",
    "ROS", "Robotics", "SLAM", "Computer Vision",
    "Image Processing", "Video Processing",
    "3D Vision", "Point Cloud", "LiDAR",
}

# NLP/IR skills that exempt someone from the offdomain penalty
NLP_IR_SKILLS = {
    "NLP", "Natural Language Processing", "Information Retrieval",
    "Text Mining", "Text Classification", "Named Entity Recognition",
    "Sentiment Analysis", "Topic Modeling", "Word Embeddings",
    "Transformers", "BERT", "GPT", "LLM", "Language Models",
    "Semantic Search", "Search", "Ranking", "Recommendation Systems",
    "RAG", "Vector Search", "Elasticsearch", "OpenSearch",
}

# ============================================================================
# Title Classification
# ============================================================================

# Non-technical titles that are strong negative signals
NON_TECH_TITLES = {
    "Marketing Manager", "HR Manager", "Operations Manager",
    "Sales Executive", "Accountant", "Content Writer",
    "Graphic Designer", "Customer Support", "Business Analyst",
    "Civil Engineer", "Mechanical Engineer", "Project Manager",
}

# AI/ML engineering titles — strong positive signal
AI_ENGINEERING_TITLES = {
    "ML Engineer", "Machine Learning Engineer",
    "Senior Machine Learning Engineer", "Staff Machine Learning Engineer",
    "AI Engineer", "Senior AI Engineer", "Lead AI Engineer",
    "AI Research Engineer", "AI Specialist",
    "Applied ML Engineer", "NLP Engineer", "Senior NLP Engineer",
    "Data Scientist", "Senior Data Scientist",
    "Search Engineer", "Ranking Engineer", "MLOps Engineer",
    "Senior Software Engineer (ML)",
}

# Software engineering titles — moderate positive
SOFTWARE_ENGINEERING_TITLES = {
    "Software Engineer", "Senior Software Engineer",
    "Full Stack Developer", "Backend Engineer",
    "Frontend Engineer", "DevOps Engineer",
    "Cloud Engineer", "Data Engineer", "Senior Data Engineer",
    "Analytics Engineer", "Data Analyst",
    "QA Engineer", "Java Developer", ".NET Developer",
    "Mobile Developer", "Junior ML Engineer",
}

# ============================================================================
# IT Services Companies (JD: career entirely at these → disqualifier)
# ============================================================================
IT_SERVICES_COMPANIES = {
    "TCS", "Infosys", "Wipro", "Accenture", "Cognizant", "Capgemini",
    "Mindtree", "HCL", "Tech Mahindra", "Mphasis", "L&T Infotech",
    "LTIMindtree", "Hexaware", "NIIT Technologies", "Cyient",
    "Persistent Systems", "Zensar",
}

# ============================================================================
# Location Scoring
# ============================================================================
# JD: Pune/Noida preferred, Hyderabad/Mumbai/Delhi NCR acceptable
PREFERRED_LOCATIONS = {
    "Pune", "Noida",
}
ACCEPTABLE_LOCATIONS = {
    "Hyderabad", "Mumbai", "Delhi", "Delhi NCR", "Gurgaon", "Gurugram",
    "New Delhi", "Bangalore", "Bengaluru", "Chennai",
}
# Broader India locations → lower score but still in scope
INDIA_LOCATIONS = {"India"}  # matched against country field

# ============================================================================
# Experience Range
# ============================================================================
YOE_IDEAL_MIN = 5.0
YOE_IDEAL_MAX = 9.0

# ============================================================================
# Behavioral Signal Thresholds
# ============================================================================
LAST_ACTIVE_RECENT_DAYS = 30
LAST_ACTIVE_MODERATE_DAYS = 90
LAST_ACTIVE_STALE_DAYS = 180
REFERENCE_DATE = datetime(2026, 6, 15)  # approximate "now" for the dataset

RESPONSE_TIME_FAST_HOURS = 24
RESPONSE_TIME_MODERATE_HOURS = 72
RESPONSE_TIME_SLOW_HOURS = 168

# ============================================================================
# Honeypot Thresholds
# ============================================================================
EXPERT_ZERO_DUR_THRESHOLD = 1  # any expert skill with 0 months → flag
CAREER_YOE_BUFFER_MONTHS = 24  # allow 2 years slack before flagging
DURATION_ELAPSED_BUFFER_MONTHS = 6  # allow 6 months slack
ASSESSMENT_MISMATCH_THRESHOLD = 30  # expert/advanced skill scored <30 on assessment
ENDORSEMENT_HIGH_PERCENTILE = 200  # very high endorsements with low completeness
COMPLETENESS_LOW_THRESHOLD = 25  # profile completeness below 25%
SALARY_MIN_FOR_SENIOR = 5.0  # LPA — unreasonably low for claimed senior
SALARY_MAX_FOR_ENTRY = 80.0  # LPA — unreasonably high for entry-level

# ============================================================================
# Production Experience Keywords
# ============================================================================
PRODUCTION_KEYWORDS = {
    "production", "deployed", "shipped", "live", "users",
    "scale", "real-time", "a/b test", "pipeline", "serving",
    "latency", "throughput", "monitoring", "reliability",
    "incident", "on-call", "sla", "traffic", "customer-facing",
    "revenue", "million", "thousands of",
}

RETRIEVAL_DOMAIN_KEYWORDS = {
    "ranking", "retrieval", "search", "recommendation",
    "embedding", "vector", "hybrid search", "semantic search",
    "information retrieval", "candidate matching",
    "relevance", "re-ranking", "bm25", "tf-idf",
    "inverted index", "approximate nearest neighbor",
    "faiss", "annoy", "hnsw",
}

# ============================================================================
# Precomputed Artifact Paths
# ============================================================================
PRECOMPUTED_DIR = "precomputed"
EMBEDDINGS_FILE = "candidate_embeddings.npy"
CANDIDATE_IDS_FILE = "candidate_ids.npy"
FEATURES_FILE = "structured_features.npz"
FAISS_INDEX_FILE = "faiss_index.bin"
COMPANY_STARTS_FILE = "company_earliest_starts.json"
JD_EMBEDDING_FILE = "jd_embedding.npy"
