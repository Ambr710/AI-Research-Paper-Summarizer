"""
app.py
------
Streamlit web interface for the AI-Based Research Paper Summarization
and Question Answering System.

Run with:
    streamlit run app.py
"""

import streamlit as st

from pdf_utils import extract_text_from_pdf, get_pdf_metadata
from preprocessing import preprocess_pipeline
from summarizer import summarize, extract_key_contributions
from qa_engine import answer_question, compare_prompt_strategies
from llm_client import is_llm_configured, which_provider

st.set_page_config(page_title="Research Paper Summarizer & Q&A", layout="wide")

st.title("📄 AI-Based Research Paper Summarization & Q&A System")
st.caption(
    "Upload a research paper (PDF) to get an automatic summary, key contributions, "
    "and ask it questions. Works fully offline by default; plug in an LLM API key "
    "for generative (abstractive) summaries and answers."
)

# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    llm_ready = is_llm_configured()
    if llm_ready:
        st.success(f"LLM configured: **{which_provider()}**")
    else:
        st.info(
            "No LLM API key found. Running in **free/offline mode** "
            "(extractive summarization + TF-IDF Q&A).\n\n"
            "To enable generative LLM mode, add `OPENAI_API_KEY` or "
            "`ANTHROPIC_API_KEY` to a `.env` file and restart."
        )

    use_llm = st.checkbox("Use LLM (if configured)", value=llm_ready, disabled=not llm_ready)
    num_summary_sentences = st.slider("Extractive summary length (sentences)", 3, 10, 6)

    st.divider()
    st.markdown("**Prompt Engineering Comparison**")
    st.caption("Compares Zero-shot / Few-shot / Chain-of-Thought prompting on the same question (requires LLM).")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "processed" not in st.session_state:
    st.session_state.processed = None

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    if st.session_state.processed is None or st.session_state.get("filename") != uploaded_file.name:
        with st.spinner("Extracting text and running NLP preprocessing..."):
            raw_text = extract_text_from_pdf(file_bytes)
            meta = get_pdf_metadata(file_bytes)
            prep = preprocess_pipeline(raw_text)

            st.session_state.processed = {"meta": meta, "prep": prep}
            st.session_state.filename = uploaded_file.name

    meta = st.session_state.processed["meta"]
    prep = st.session_state.processed["prep"]

    st.success(f"Extracted **{prep['num_words']}** words across **{meta['num_pages']}** pages "
               f"({prep['num_sentences']} usable sentences after cleaning).")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Summary", "🔑 Key Contributions & Keywords", "❓ Ask a Question", "🧪 Prompt Strategy Comparison"]
    )

    # --- Tab 1: Summary ---
    with tab1:
        if st.button("Generate Summary", key="gen_summary"):
            with st.spinner("Summarizing..."):
                result = summarize(
                    prep["sentences"], prep["cleaned_text"],
                    use_llm=use_llm, num_sentences=num_summary_sentences,
                )
            st.markdown(f"**Method used:** `{result['method']}`")
            if result["abstractive_summary"]:
                st.subheader("Abstractive Summary (LLM-generated)")
                st.write(result["abstractive_summary"])
                with st.expander("See extractive summary too"):
                    st.write(result["extractive_summary"])
            else:
                st.subheader("Extractive Summary")
                st.write(result["extractive_summary"])

    # --- Tab 2: Key contributions & keywords ---
    with tab2:
        if st.button("Extract Key Contributions & Keywords", key="gen_keywords"):
            with st.spinner("Analyzing..."):
                contributions = extract_key_contributions(prep["sentences"], prep["keywords"])
            st.subheader("Top Keywords")
            st.write(", ".join(prep["keywords"]))
            st.subheader("Likely Key Contributions")
            for i, c in enumerate(contributions, 1):
                st.markdown(f"{i}. {c}")

    # --- Tab 3: Q&A ---
    with tab3:
        question = st.text_input("Ask a question about this paper:")
        if st.button("Get Answer", key="get_answer") and question:
            with st.spinner("Finding answer..."):
                result = answer_question(question, prep["sentences"], use_llm=use_llm)
            st.markdown(f"**Method used:** `{result['method']}`")
            st.write(result["answer"])

    # --- Tab 4: Prompt strategy comparison ---
    with tab4:
        if not llm_ready:
            st.warning("This feature requires an LLM API key (OpenAI or Anthropic). Add one to `.env` to enable it.")
        else:
            compare_question = st.text_input("Question to compare across prompting strategies:", key="cmp_q")
            if st.button("Compare Strategies", key="cmp_btn") and compare_question:
                with st.spinner("Running all three prompting strategies..."):
                    results = compare_prompt_strategies(compare_question, prep["sentences"])
                cols = st.columns(3)
                titles = {
                    "zero_shot": "Zero-Shot",
                    "few_shot": "Few-Shot",
                    "chain_of_thought": "Chain-of-Thought",
                }
                for col, (key, title) in zip(cols, titles.items()):
                    with col:
                        st.markdown(f"**{title}**")
                        st.write(results[key])
else:
    st.info("👆 Upload a PDF research paper to get started.")
