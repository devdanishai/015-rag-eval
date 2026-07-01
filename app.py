"""
app.py  —  RAG Evaluation App  (Streamlit UI)
─────────────────────────────────────────────
Run:
    streamlit run app.py

Pages:
  1. Ask the Bot     — live RAG Q&A with retrieved chunks
  2. Run Evaluation  — RAGAS scoring over N samples
  3. Results History — browse past eval runs
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
RESULTS_DIR = ROOT / "results"

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Eval App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    border: 1px solid #313244;
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #cba6f7; }
.metric-label { font-size: 0.85rem; color: #a6adc8; margin-top: 4px; }
.pass-badge  { background:#a6e3a1; color:#1e1e2e; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.fail-badge  { background:#f38ba8; color:#1e1e2e; padding:2px 10px; border-radius:20px; font-size:0.8rem; font-weight:600; }
.chunk-box {
    background: #181825;
    border-left: 3px solid #89b4fa;
    border-radius: 6px;
    padding: 12px 16px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    color: #cdd6f4;
}
</style>
""", unsafe_allow_html=True)


# ── helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


VENV_PYTHON = str(ROOT / ".venv" / "bin" / "python")


def rag_query(question: str) -> dict:
    """
    Run RAG via subprocess — uses full GPU venv, no model in Streamlit process.
    """
    import json as _json
    result = subprocess.run(
        [VENV_PYTHON, "src/query_cli.py", question],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-500:])
    return _json.loads(result.stdout.strip())


def load_results() -> list[dict]:
    runs = []
    for run_dir in sorted(RESULTS_DIR.glob("*/"), reverse=True):
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                runs.append(json.load(f))
    return runs


def score_color(score: float, threshold: float) -> str:
    if score >= threshold + 0.1:
        return "#a6e3a1"
    elif score >= threshold:
        return "#f9e2af"
    return "#f38ba8"


def progress_bar_html(score: float, color: str) -> str:
    pct = int(score * 100)
    return (
        f'<div style="background:#313244;border-radius:6px;height:10px;'
        f'width:100%;margin-top:6px">'
        f'<div style="background:{color};border-radius:6px;height:10px;width:{pct}%"></div>'
        f'</div>'
        f'<div style="font-size:0.75rem;color:#a6adc8;margin-top:3px">{score:.4f}</div>'
    )


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search.png", width=60)
    st.title("RAG Eval App")
    st.caption("Self-hosted · Qwen 14B · RAGAS")
    st.divider()

    page = st.radio(
        "Navigate",
        ["💬 Ask the Bot", "📊 Run Evaluation", "📁 Results History"],
        label_visibility="collapsed",
    )

    st.divider()
    cfg = load_config()
    st.caption(f"**Model:** `{cfg['llm']['app_model']['model'].split('/')[-1]}`")
    st.caption(f"**Embeddings:** `{cfg['embeddings']['model'].split('/')[-1]}`")
    st.caption(f"**Top-k:** `{cfg['retrieval']['top_k']}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Ask the Bot
# ══════════════════════════════════════════════════════════════════════════════
if page == "💬 Ask the Bot":
    st.title("💬 Ask the Support Bot")
    st.caption("Powered by Qwen 14B AWQ · Qdrant · BGE-M3")

    question = st.text_input(
        "Your question",
        placeholder="e.g. How do I cancel my subscription?",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 5])
    ask_btn = col1.button("Ask", type="primary", use_container_width=True)
    col2.empty()

    if ask_btn and question.strip():
        with st.spinner("Retrieving & generating…"):
            try:
                t0 = time.time()
                result = rag_query(question)
                elapsed = time.time() - t0
            except Exception as e:
                st.error(f"Could not connect. Make sure vLLM (port 8000) and Qdrant (port 6333) are running.\n\n`{e}`")
                st.stop()

        st.success(f"Answer  ·  _{elapsed:.1f}s_")
        st.markdown(f"### {result['answer']}")

        st.divider()
        st.markdown(f"**Retrieved {len(result['contexts'])} context chunks**")

        for i, chunk in enumerate(result["contexts"], 1):
            st.markdown(
                f'<div class="chunk-box"><strong>Chunk {i}</strong><br>{chunk[:400]}{"…" if len(chunk)>400 else ""}</div>',
                unsafe_allow_html=True,
            )

    elif ask_btn:
        st.warning("Please enter a question.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Run Evaluation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Run Evaluation":
    st.title("📊 Run RAGAS Evaluation")
    st.caption("Scores: Faithfulness · Context Precision · Context Recall")

    with st.form("eval_form"):
        c1, c2 = st.columns(2)
        n_samples = c1.slider("Number of samples", 5, 100, 20, step=5)
        tag = c2.text_input("Run tag", placeholder="e.g. v2-new-prompt")
        submitted = st.form_submit_button("▶ Run Evaluation", type="primary")

    if submitted:
        tag_arg = tag.strip() or "run"
        cmd = [
            sys.executable, "-m", "src.eval_runner",
            "--samples", str(n_samples),
            "--tag", tag_arg,
        ]

        st.info(f"Running eval on {n_samples} samples… this takes ~{n_samples} minutes.")
        progress = st.progress(0, text="Starting…")
        log_area = st.empty()
        logs = []

        with st.spinner("Evaluating…"):
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            step = 0
            for line in proc.stdout:
                logs.append(line.rstrip())
                log_area.code("\n".join(logs[-15:]), language="bash")
                if "/" in line and "Running RAG" in line:
                    try:
                        cur, total = line.split("[")[1].split("]")[0].split("/")
                        progress.progress(int(cur) / int(total), text=f"Sample {cur}/{total}")
                        step = int(cur)
                    except Exception:
                        pass
            proc.wait()

        if proc.returncode == 0:
            progress.progress(1.0, text="Done!")
            st.success("Evaluation complete! See Results History for scores.")

            # show latest result inline
            runs = load_results()
            if runs:
                latest = runs[0]
                scores = latest.get("scores", {})
                flags = latest.get("threshold_flags", {})

                st.divider()
                st.subheader(f"Results — {latest.get('tag', '')}")
                cols = st.columns(len(scores))
                for col, (metric, score) in zip(cols, scores.items()):
                    info = flags.get(metric, {})
                    color = score_color(score, info.get("threshold", 0.7))
                    badge = "pass-badge" if info.get("passed") else "fail-badge"
                    status = "PASS" if info.get("passed") else "FAIL"
                    col.markdown(
                        f"""<div class="metric-card">
                        <div class="metric-value" style="color:{color}">{score:.2f}</div>
                        <div class="metric-label">{metric.replace("_"," ").title()}</div>
                        <div style="margin-top:8px"><span class="{badge}">{status}</span></div>
                        </div>""",
                        unsafe_allow_html=True,
                    )
        else:
            st.error("Evaluation failed. Check logs above.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Results History
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Results History":
    st.title("📁 Results History")

    runs = load_results()

    if not runs:
        st.info("No eval runs yet. Go to **Run Evaluation** to create one.")
        st.stop()

    # ── run selector ──────────────────────────────────────────────────────────
    run_labels = [f"{r['tag']}  ·  {r['timestamp']}  ·  {r['n_samples']} samples" for r in runs]
    selected_idx = st.selectbox("Select run", range(len(runs)), format_func=lambda i: run_labels[i])
    run = runs[selected_idx]

    st.divider()

    # ── score cards ───────────────────────────────────────────────────────────
    scores = run.get("scores", {})
    flags = run.get("threshold_flags", {})
    thresholds = {m: flags[m]["threshold"] for m in flags}

    st.subheader("Metric Scores")
    cols = st.columns(len(scores))
    for col, (metric, score) in zip(cols, scores.items()):
        info = flags.get(metric, {})
        color = score_color(score, info.get("threshold", 0.7))
        badge = "pass-badge" if info.get("passed") else "fail-badge"
        status = "PASS" if info.get("passed") else "FAIL"
        col.markdown(
            f"""<div class="metric-card">
            <div class="metric-value" style="color:{color}">{score:.4f}</div>
            <div class="metric-label">{metric.replace("_"," ").title()}</div>
            <div style="margin-top:8px"><span class="{badge}">{status}</span></div>
            {progress_bar_html(score, color)}
            </div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── run metadata ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", run.get("n_samples", "—"))
    c2.metric("Model", run.get("model", "—").split("/")[-1])
    c3.metric("Judge", run.get("judge", "—").split("/")[-1])

    st.divider()

    # ── per-question table ────────────────────────────────────────────────────
    run_dir = RESULTS_DIR / run["run_id"]
    csv_path = run_dir / "per_question.csv"

    if csv_path.exists():
        st.subheader("Per-Question Results")
        df = pd.read_csv(csv_path)

        metric_cols = [c for c in df.columns if c in ["faithfulness", "context_precision", "context_recall"]]
        display_cols = ["question", "answer"] + metric_cols
        available = [c for c in display_cols if c in df.columns]
        df_show = df[available].copy()

        # color metric cells
        def color_score(val):
            try:
                v = float(val)
                if v >= 0.8:
                    return "background-color:#1a3a2a; color:#a6e3a1"
                elif v >= 0.6:
                    return "background-color:#3a3020; color:#f9e2af"
                return "background-color:#3a1a1a; color:#f38ba8"
            except Exception:
                return ""

        styled = df_show.style.map(color_score, subset=metric_cols if metric_cols else [])
        st.dataframe(styled, use_container_width=True, height=400)

        # download button
        st.download_button(
            "⬇ Download CSV",
            data=df.to_csv(index=False),
            file_name=f"eval_{run['run_id']}.csv",
            mime="text/csv",
        )
    else:
        st.warning("Per-question CSV not found for this run.")

    st.divider()

    # ── compare with previous run ─────────────────────────────────────────────
    if len(runs) > 1:
        st.subheader("Compare with Another Run")
        compare_idx = st.selectbox(
            "Compare against",
            [i for i in range(len(runs)) if i != selected_idx],
            format_func=lambda i: run_labels[i],
        )
        compare_run = runs[compare_idx]

        compare_data = []
        all_metrics = list(scores.keys())
        for m in all_metrics:
            a = scores.get(m, 0)
            b = compare_run.get("scores", {}).get(m, 0)
            delta = a - b
            compare_data.append({
                "Metric": m.replace("_", " ").title(),
                run.get("tag", "Selected"): round(a, 4),
                compare_run.get("tag", "Compare"): round(b, 4),
                "Delta": f"{'+' if delta >= 0 else ''}{delta:.4f}",
            })

        st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)
