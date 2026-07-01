"""
rag_pipeline.py
───────────────
LangChain RAG pipeline using:
  - Qdrant as vector store
  - BGE-M3 for query embedding
  - Qwen 14B AWQ (via vLLM) as the generation LLM
  - Langfuse for tracing every request

Usage (as a module):
    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    result = pipeline.query("How do I cancel my subscription?")
    print(result["answer"])
    print(result["contexts"])

Usage (standalone test):
    python src/rag_pipeline.py --question "How do I cancel my subscription?"
"""

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"

SYSTEM_PROMPT = """You are a helpful customer support assistant.
Answer the customer's question using ONLY the information provided in the context below.
If the context does not contain enough information to answer, say exactly:
"I don't have that information. Please contact our support team directly."

Do NOT make up policies, prices, or procedures.
Keep your answer concise and friendly.

Context:
{context}"""

HUMAN_PROMPT = "{question}"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class RAGPipeline:
    def __init__(self, cfg: dict | None = None, tracer=None):
        self.cfg = cfg or load_config()
        self.tracer = tracer  # optional Langfuse tracer

        self._init_embedder()
        self._init_qdrant()
        self._init_llm()
        self._build_chain()

    def _init_embedder(self):
        emb_cfg = self.cfg["embeddings"]
        self.embedder = SentenceTransformer(
            emb_cfg["model"], device=emb_cfg["device"]
        )

    def _init_qdrant(self):
        self.qdrant = QdrantClient(url=self.cfg["qdrant"]["url"])
        self.collection = self.cfg["qdrant"]["collection_name"]

    def _init_llm(self):
        llm_cfg = self.cfg["llm"]["app_model"]
        self.llm = ChatOpenAI(
            base_url=llm_cfg["base_url"],
            api_key="not-needed",
            model=llm_cfg["model"],
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
        )

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
        )
        self.chain = (
            RunnablePassthrough()
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def retrieve(self, question: str) -> list[str]:
        """Embed the question and retrieve top-k chunks from Qdrant."""
        ret_cfg = self.cfg["retrieval"]
        query_vec = self.embedder.encode(
            question, normalize_embeddings=True
        ).tolist()

        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=ret_cfg["top_k"],
            score_threshold=ret_cfg["score_threshold"],
        ).points
        return [hit.payload.get("page_content", "") for hit in results]

    def query(self, question: str) -> dict:
        """
        Run the full RAG pipeline.

        Returns:
            {
                "question": str,
                "contexts": list[str],   # retrieved chunks
                "answer": str,
            }
        """
        span = None
        if self.tracer:
            span = self.tracer.start_span(name="rag_query", input=question)

        contexts = self.retrieve(question)
        context_text = "\n\n---\n\n".join(contexts) if contexts else "No context found."

        answer = self.chain.invoke(
            {"context": context_text, "question": question}
        )

        if span:
            self.tracer.end_span(
                span,
                output=answer,
                metadata={"num_contexts": len(contexts)},
            )

        return {
            "question": question,
            "contexts": contexts,
            "answer": answer,
        }


def main():
    parser = argparse.ArgumentParser(description="Test RAG pipeline with a single question")
    parser.add_argument(
        "--question",
        type=str,
        default="How do I cancel my subscription?",
        help="Question to ask the support bot",
    )
    args = parser.parse_args()

    pipeline = RAGPipeline()
    result = pipeline.query(args.question)

    print("\n" + "=" * 60)
    print(f"Question:  {result['question']}")
    print("=" * 60)
    print(f"Answer:\n{result['answer']}")
    print("=" * 60)
    print(f"Retrieved {len(result['contexts'])} context chunks.")
    for i, ctx in enumerate(result["contexts"], 1):
        print(f"\n[Chunk {i}]\n{ctx[:300]}...")


if __name__ == "__main__":
    main()
