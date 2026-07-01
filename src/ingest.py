"""
ingest.py
─────────
Load the Bitext customer-support dataset from HuggingFace,
chunk the texts, embed with BGE-M3, and upsert into Qdrant.

Usage:
    python src/ingest.py               # uses config.yaml defaults
    python src/ingest.py --sample 200  # override sample size
"""

import argparse
import os
import uuid
from pathlib import Path

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_support_dataset(cfg: dict) -> list[Document]:
    """
    Load Bitext customer-support dataset and convert rows to LangChain Documents.
    Each row has: instruction (question) + response (answer) + category + intent.
    We store the response as page_content and metadata on the doc.
    """
    ds_cfg = cfg["dataset"]
    print(f"Loading dataset: {ds_cfg['hf_name']}")

    ds = load_dataset(
        ds_cfg["hf_name"],
        split=ds_cfg["split"],
        cache_dir=ds_cfg["cache_dir"],
    )

    sample_size = ds_cfg.get("sample_size")
    if sample_size:
        ds = ds.select(range(min(sample_size, len(ds))))

    print(f"Loaded {len(ds)} rows. Converting to Documents...")

    docs = []
    for row in ds:
        # Combine question + answer as the indexed text so retrieval can
        # match on both the question pattern and the answer content.
        text = f"Question: {row['instruction']}\nAnswer: {row['response']}"
        doc = Document(
            page_content=text,
            metadata={
                "question": row["instruction"],
                "answer": row["response"],
                "category": row.get("category", ""),
                "intent": row.get("intent", ""),
                "source": ds_cfg["hf_name"],
            },
        )
        docs.append(doc)

    return docs


def chunk_documents(docs: list[Document], cfg: dict) -> list[Document]:
    chunk_cfg = cfg["chunking"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
    )
    chunks = splitter.split_documents(docs)
    print(f"Chunked into {len(chunks)} pieces.")
    return chunks


def embed_and_upsert(chunks: list[Document], cfg: dict) -> None:
    emb_cfg = cfg["embeddings"]
    qdrant_cfg = cfg["qdrant"]

    print(f"Loading embedding model: {emb_cfg['model']}")
    model = SentenceTransformer(emb_cfg["model"], device=emb_cfg["device"])

    client = QdrantClient(url=qdrant_cfg["url"])
    collection = qdrant_cfg["collection_name"]

    # (Re)create collection
    existing = [c.name for c in client.get_collections().collections]
    if collection in existing:
        print(f"Dropping existing collection '{collection}'...")
        client.delete_collection(collection)

    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(
            size=qdrant_cfg["vector_size"],
            distance=Distance.COSINE,
        ),
    )
    print(f"Created Qdrant collection '{collection}'.")

    batch_size = emb_cfg["batch_size"]
    texts = [c.page_content for c in chunks]
    points = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i : i + batch_size]
        batch_chunks = chunks[i : i + batch_size]
        vectors = model.encode(batch_texts, normalize_embeddings=True).tolist()

        for vec, chunk in zip(vectors, batch_chunks):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "page_content": chunk.page_content,
                        **chunk.metadata,
                    },
                )
            )

    client.upsert(collection_name=collection, points=points)
    print(f"Upserted {len(points)} vectors into Qdrant '{collection}'.")


def main():
    parser = argparse.ArgumentParser(description="Ingest support dataset into Qdrant")
    parser.add_argument("--sample", type=int, help="Override sample_size from config")
    args = parser.parse_args()

    cfg = load_config()
    if args.sample:
        cfg["dataset"]["sample_size"] = args.sample

    docs = load_support_dataset(cfg)
    chunks = chunk_documents(docs, cfg)
    embed_and_upsert(chunks, cfg)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
