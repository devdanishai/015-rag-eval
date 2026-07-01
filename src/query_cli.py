"""
query_cli.py
────────────
Called by Streamlit as a subprocess. Runs the full RAG pipeline
(CPU embeddings + vLLM HTTP) and prints JSON result to stdout.

Usage:
    python src/query_cli.py "How do I cancel my subscription?"
"""

import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    cfg = yaml.safe_load(f)

# BGE-M3 on CPU — vLLM already occupies the full GPU
cfg["embeddings"]["device"] = "cpu"

from src.rag_pipeline import RAGPipeline

question = sys.argv[1] if len(sys.argv) > 1 else "Hello"
pipeline = RAGPipeline(cfg=cfg)
result = pipeline.query(question)
print(json.dumps(result))
