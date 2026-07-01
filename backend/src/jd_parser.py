"""
JD Parser — converts job_description.docx text into a structured signal dict.
Used by all scoring lanes. Run once, cache to data/jd_signals.json.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Set

@dataclass
class JDSignals:
    # Role identity
    title: str = "Senior AI Engineer"
    company_type: str = "series_a_product"
    location_preferred: List[str] = field(default_factory=lambda: ["pune", "noida", "hyderabad", "mumbai", "delhi", "bengaluru", "bangalore"])
    yoe_min: float = 4.0
    yoe_max: float = 11.0
    yoe_sweet_spot: tuple = (5, 9)

    # Hard must-haves (any candidate missing ALL of these → low score)
    must_have_skills: List[str] = field(default_factory=lambda: [
        "embeddings", "vector database", "retrieval", "semantic search",
        "sentence transformers", "faiss", "pinecone", "weaviate", "qdrant",
        "milvus", "elasticsearch", "opensearch", "dense retrieval",
        "ranking", "recommendation system", "information retrieval",
        "ndcg", "mrr", "a/b testing", "python"
    ])

    # Nice-to-haves (positive multiplier if present)
    nice_to_have_skills: List[str] = field(default_factory=lambda: [
        "lora", "qlora", "peft", "fine-tuning", "lightgbm", "xgboost",
        "learning to rank", "open source", "github", "distributed systems",
        "hr tech", "recruiting", "marketplace", "bm25", "hybrid search",
        "reranker", "cross encoder"
    ])

    # Career description keywords (from actual work descriptions, not just skill tags)
    career_positive_keywords: List[str] = field(default_factory=lambda: [
        "production", "deployed", "shipped", "real users", "at scale",
        "embedding", "retrieval", "search", "ranking", "recommendation",
        "vector", "similarity", "index", "recall", "precision",
        "a/b test", "offline eval", "online eval", "latency",
        "encoder", "bert", "transformer", "nlp", "rag"
    ])

    # Explicit disqualifiers from the JD (hard kills)
    disqualifier_titles: List[str] = field(default_factory=lambda: [
        "marketing", "sales", "hr", "human resources", "finance",
        "operations", "project manager", "business analyst", "content writer",
        "seo", "digital marketing", "product marketing"
    ])

    disqualifier_companies: List[str] = field(default_factory=lambda: [
        "tcs", "tata consultancy", "infosys", "wipro", "accenture",
        "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
        "l&t infotech", "hexaware", "mindtree", "happiest minds"
    ])

    # The JD explicitly says these are NOT wanted
    anti_patterns: List[str] = field(default_factory=lambda: [
        "langchain tutorial", "openai wrapper", "chatgpt demo",
        "computer vision only", "speech recognition only", "robotics only",
        "research only", "no production", "academic"
    ])

    # What the JD actually means (read between the lines)
    # "product company experience" is the single most important signal
    preferred_company_types: List[str] = field(default_factory=lambda: [
        "startup", "series", "product", "saas", "marketplace",
        "ecommerce", "fintech", "edtech", "healthtech", "proptech",
        "ai", "ml", "data"
    ])

    # Location scoring
    india_tier1_cities: List[str] = field(default_factory=lambda: [
        "pune", "noida", "hyderabad", "mumbai", "delhi", "bengaluru",
        "bangalore", "gurugram", "gurgaon", "chennai"
    ])


def parse_jd() -> JDSignals:
    """
    Returns the pre-defined JD signals for the Redrob Senior AI Engineer role.
    In a real system, this would use an LLM to parse arbitrary JDs.
    For this challenge, we hard-code because the JD is fixed.
    """
    return JDSignals()


def save_jd_signals(path: str = "data/jd_signals.json"):
    Path("data").mkdir(exist_ok=True)
    signals = parse_jd()
    with open(path, "w") as f:
        json.dump(asdict(signals), f, indent=2)
    print(f"[JD Parser] Saved signals to {path}")
    return signals


if __name__ == "__main__":
    save_jd_signals()
