"""
eval_runner.py
──────────────
Run RAGAS evaluation over a sample of the Bitext dataset.

Steps:
  1. Load N questions from the dataset (ground_truth answers already there)
  2. Run each question through the RAG pipeline (retrieve + generate)
  3. Score with RAGAS metrics using the judge LLM (Qwen 7B)
  4. Save results to results/eval_<timestamp>.json and a summary CSV

Usage:
    python src/eval_runner.py                    # default 50 samples
    python src/eval_runner.py --samples 20       # quick test run
    python src/eval_runner.py --samples 100 --tag "v2-prompt"
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, context_precision, context_recall
from datasets import Dataset as HFDataset

from src.rag_pipeline import RAGPipeline

load_dotenv()

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_judge_llm(cfg: dict) -> LangchainLLMWrapper:
    """Wrap the vLLM endpoint as a RAGAS LangchainLLMWrapper judge."""
    judge_cfg = cfg["llm"]["judge_model"]
    lc_llm = ChatOpenAI(
        base_url=judge_cfg["base_url"],
        api_key="not-needed",
        model=judge_cfg["model"],
        temperature=judge_cfg["temperature"],
        max_tokens=judge_cfg["max_tokens"],
    )
    return LangchainLLMWrapper(lc_llm)



def collect_eval_samples(
    pipeline: RAGPipeline,
    cfg: dict,
    n_samples: int,
) -> HFDataset:
    """
    For each row in the dataset:
      - run RAG pipeline to get contexts + generated answer
      - return as HuggingFace Dataset (old RAGAS format)
    """
    ds_cfg = cfg["dataset"]
    ds = load_dataset(
        ds_cfg["hf_name"],
        split=ds_cfg["split"],
        cache_dir=ds_cfg["cache_dir"],
    )
    ds = ds.select(range(min(n_samples, len(ds))))

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, row in enumerate(ds):
        question = row["instruction"]
        ground_truth = row["response"]

        print(f"[{i+1}/{len(ds)}] Running RAG for: {question[:70]}...")
        result = pipeline.query(question)

        questions.append(question)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(ground_truth)

    return HFDataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })


def score_thresholds(scores: dict, cfg: dict) -> dict:
    """Compare each metric score against configured thresholds."""
    thresholds = cfg["eval"]["thresholds"]
    flags = {}
    for metric, threshold in thresholds.items():
        val = scores.get(metric)
        if val is not None:
            flags[metric] = {
                "score": round(float(val), 4),
                "threshold": threshold,
                "passed": float(val) >= threshold,
            }
    return flags


def save_results(
    scores: dict,
    threshold_flags: dict,
    raw_df: pd.DataFrame,
    tag: str,
    cfg: dict,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = f"{ts}_{tag}" if tag else ts
    out_dir = RESULTS_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        "run_id": label,
        "timestamp": ts,
        "tag": tag,
        "model": cfg["llm"]["app_model"]["model"],
        "judge": cfg["llm"]["judge_model"]["model"],
        "n_samples": len(raw_df),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
        "threshold_flags": threshold_flags,
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Per-question CSV
    csv_path = out_dir / "per_question.csv"
    raw_df.to_csv(csv_path, index=False)

    print(f"\nResults saved to: {out_dir}")
    return out_dir


def print_report(scores: dict, threshold_flags: dict, n: int):
    print("\n" + "=" * 60)
    print(f"  RAGAS Evaluation Report  ({n} questions)")
    print("=" * 60)
    for metric, info in threshold_flags.items():
        status = "PASS" if info["passed"] else "FAIL"
        bar = "█" * int(info["score"] * 20)
        print(
            f"  {metric:<22} {info['score']:.4f}  [{bar:<20}]  "
            f"(threshold: {info['threshold']})  [{status}]"
        )
    print("=" * 60)
    all_passed = all(v["passed"] for v in threshold_flags.values())
    verdict = "ALL METRICS PASSED" if all_passed else "SOME METRICS BELOW THRESHOLD"
    print(f"  Overall: {verdict}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on the RAG pipeline")
    parser.add_argument("--samples", type=int, default=50, help="Number of questions to evaluate")
    parser.add_argument("--tag", type=str, default="", help="Label for this run (e.g. v2-prompt)")
    args = parser.parse_args()

    cfg = load_config()

    print("Initializing RAG pipeline...")
    pipeline = RAGPipeline(cfg=cfg)

    print("Building RAGAS judge (Qwen 14B via vLLM)...")
    judge_llm = build_judge_llm(cfg)
    print(f"Collecting {args.samples} eval samples...")
    samples = collect_eval_samples(pipeline, cfg, args.samples)

    faithfulness.llm = judge_llm
    context_precision.llm = judge_llm
    context_recall.llm = judge_llm

    metrics = [faithfulness, context_precision, context_recall]

    run_cfg = RunConfig(
        max_workers=1,       # one judge call at a time — single GPU
        timeout=300,         # 5 min per call
        max_retries=2,
    )

    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        dataset=samples,
        metrics=metrics,
        run_config=run_cfg,
        raise_exceptions=False,
    )

    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    raw_df = result.to_pandas()

    threshold_flags = score_thresholds(scores, cfg)
    print_report(scores, threshold_flags, n=args.samples)
    save_results(scores, threshold_flags, raw_df, args.tag, cfg)


if __name__ == "__main__":
    main()
