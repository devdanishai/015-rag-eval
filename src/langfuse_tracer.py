"""
langfuse_tracer.py
──────────────────
Thin wrapper around Langfuse for tracing RAG queries.
Traces every query with:
  - input question
  - retrieved contexts
  - generated answer
  - latency
  - RAGAS scores (optional, logged after eval)

Usage:
    from src.langfuse_tracer import LangfuseTracer
    tracer = LangfuseTracer(cfg)

    with tracer.trace("rag_query", input="How do I cancel?") as span:
        result = pipeline.query(question)
        span.update(output=result["answer"], metadata={"n_contexts": 5})

    # Log eval scores back to the trace
    tracer.log_scores(trace_id, {"faithfulness": 0.91, "answer_relevancy": 0.88})
"""

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class _NoOpSpan:
    """Returned when Langfuse is disabled so callers don't need to check."""

    def __init__(self):
        self.id = "noop"

    def update(self, **kwargs):
        pass

    def end(self, **kwargs):
        pass


class LangfuseTracer:
    """
    Wraps Langfuse tracing. Gracefully degrades to no-ops when:
      - langfuse.enabled = false in config.yaml
      - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set
      - Langfuse server is unreachable
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self._client = None
        self._enabled = False
        self._init_client()

    def _init_client(self):
        lf_cfg = self.cfg.get("langfuse", {})
        if not lf_cfg.get("enabled", False):
            print("[Langfuse] Tracing disabled via config.")
            return

        pub_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        sec_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        host = os.getenv("LANGFUSE_HOST", lf_cfg.get("host", "http://localhost:3000"))

        if not pub_key or not sec_key:
            print("[Langfuse] Keys not set in .env — tracing disabled.")
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=pub_key,
                secret_key=sec_key,
                host=host,
            )
            self._enabled = True
            print(f"[Langfuse] Tracing enabled → {host}")
        except Exception as e:
            print(f"[Langfuse] Failed to connect: {e}. Tracing disabled.")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_trace(self, name: str, input: Any = None, metadata: dict | None = None):
        """Start a top-level Langfuse trace. Returns trace object or None."""
        if not self._enabled:
            return None
        return self._client.trace(name=name, input=input, metadata=metadata or {})

    def start_span(self, trace, name: str, input: Any = None):
        """Start a span inside a trace."""
        if not self._enabled or trace is None:
            return _NoOpSpan()
        return trace.span(name=name, input=input, start_time=time.time())

    def end_span(self, span, output: Any = None, metadata: dict | None = None):
        if isinstance(span, _NoOpSpan):
            return
        span.end(output=output, metadata=metadata or {})

    def log_scores(self, trace_id: str, scores: dict[str, float]):
        """
        Log RAGAS scores back onto a trace after eval.
        Call this from eval_runner after scoring.
        """
        if not self._enabled:
            return
        for name, value in scores.items():
            self._client.score(
                trace_id=trace_id,
                name=name,
                value=value,
            )

    @contextmanager
    def trace_query(self, question: str):
        """
        Context manager for a single RAG query trace.

        Usage:
            with tracer.trace_query(question) as ctx:
                result = pipeline.query(question)
                ctx["output"] = result["answer"]
                ctx["metadata"] = {"n_contexts": len(result["contexts"])}
        """
        ctx: dict[str, Any] = {"output": None, "metadata": {}, "trace_id": None}

        if not self._enabled:
            yield ctx
            return

        trace = self._client.trace(name="rag_query", input=question)
        span = trace.span(name="retrieve_and_generate", input=question)
        ctx["trace_id"] = trace.id

        try:
            yield ctx
        finally:
            span.end(output=ctx.get("output"), metadata=ctx.get("metadata", {}))
            trace.update(output=ctx.get("output"))

    def flush(self):
        """Flush pending events to Langfuse (call before process exit)."""
        if self._enabled and self._client:
            self._client.flush()
