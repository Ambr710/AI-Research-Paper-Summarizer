"""
summarizer.py
-------------
Two summarization engines:

1. Extractive TextRank (default, FREE, fully offline, no API key needed).
   Picks the most 'central' sentences from the paper using a TF-IDF +
   cosine-similarity graph, ranked with PageRank (networkx).

2. LLM abstractive summarization (optional). If the user has configured
   an OpenAI or Anthropic API key (see llm_client.py), this produces a
   genuinely 'generated' abstractive summary instead of extracted sentences.
"""

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer

from llm_client import get_llm_response, is_llm_configured


def textrank_summary(sentences: list, num_sentences: int = 5) -> str:
    """Extractive summary using TF-IDF + cosine similarity + PageRank."""
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    sim_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    np.fill_diagonal(sim_matrix, 0)

    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph, max_iter=200)

    ranked_idx = sorted(scores, key=scores.get, reverse=True)[:num_sentences]
    ranked_idx.sort()  # keep original order for readability
    return " ".join(sentences[i] for i in ranked_idx)


def llm_abstractive_summary(paper_text: str, max_chars: int = 12000) -> str:
    """Generate an abstractive summary using a configured LLM API (optional)."""
    prompt = (
        "You are an expert research assistant. Read the following research "
        "paper excerpt and write a concise, well-structured abstractive "
        "summary (150-250 words) covering: the problem addressed, the "
        "proposed method, and the main results.\n\n"
        f"PAPER TEXT:\n{paper_text[:max_chars]}"
    )
    return get_llm_response(prompt)


def summarize(sentences: list, full_text: str, use_llm: bool, num_sentences: int = 6) -> dict:
    """Unified entry point used by the Streamlit app."""
    extractive = textrank_summary(sentences, num_sentences=num_sentences)
    result = {"extractive_summary": extractive, "abstractive_summary": None, "method": "extractive"}

    if use_llm and is_llm_configured():
        result["abstractive_summary"] = llm_abstractive_summary(full_text)
        result["method"] = "abstractive (LLM)"

    return result


def extract_key_contributions(sentences: list, keywords: list, top_n: int = 5) -> list:
    """
    Heuristic extraction of 'key contribution' style sentences: sentences
    that contain contribution-signal phrases AND at least one top keyword.
    Free/offline fallback when no LLM key is configured.
    """
    signal_phrases = [
        "we propose", "we present", "we introduce", "our approach",
        "our method", "this paper", "we show", "we demonstrate",
        "contribution", "novel", "outperform", "achieve", "state-of-the-art",
    ]
    scored = []
    keyword_set = set(keywords)
    for s in sentences:
        s_lower = s.lower()
        signal_hits = sum(1 for p in signal_phrases if p in s_lower)
        keyword_hits = sum(1 for k in keyword_set if k in s_lower)
        if signal_hits > 0:
            scored.append((signal_hits * 2 + keyword_hits, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_n]] if scored else sentences[:top_n]
