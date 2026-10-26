"""
qa_engine.py
------------
Question-Answering over the uploaded paper.

Free/offline mode: TF-IDF retrieval returns the most relevant sentences
    to the question (extractive QA - no LLM needed).

LLM mode: retrieved context + question is sent to the configured LLM,
    which generates a natural-language answer. Three different prompt
    engineering strategies are implemented so you can compare them
    (this satisfies the 'compare prompt engineering strategies' objective).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_client import get_llm_response, is_llm_configured


def retrieve_relevant_sentences(question: str, sentences: list, top_k: int = 5) -> list:
    """TF-IDF + cosine similarity retrieval of the sentences most relevant to `question`."""
    if not sentences:
        return []

    corpus = sentences + [question]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus)

    question_vec = tfidf[-1]
    sentence_vecs = tfidf[:-1]
    sims = cosine_similarity(question_vec, sentence_vecs).flatten()

    ranked_idx = sims.argsort()[::-1][:top_k]
    ranked_idx = [i for i in ranked_idx if sims[i] > 0]
    return [sentences[i] for i in ranked_idx]


def extractive_answer(question: str, sentences: list) -> str:
    """Free fallback: just return the most relevant sentences as the 'answer'."""
    relevant = retrieve_relevant_sentences(question, sentences, top_k=3)
    if not relevant:
        return "I couldn't find a directly relevant passage in the paper for that question."
    return " ".join(relevant)


# ---------------------------------------------------------------------------
# Prompt engineering strategies (used only when an LLM is configured)
# ---------------------------------------------------------------------------

def _build_prompt(strategy: str, question: str, context: str) -> str:
    if strategy == "zero_shot":
        return (
            f"Context from a research paper:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer the question using only the context above."
        )

    if strategy == "few_shot":
        return (
            "Answer research-paper questions concisely and factually, using only the given context.\n\n"
            "Example:\nContext: 'We use a ResNet-50 backbone pretrained on ImageNet.'\n"
            "Question: What backbone is used?\nAnswer: A ResNet-50 backbone pretrained on ImageNet.\n\n"
            f"Now answer this one:\nContext: {context}\n"
            f"Question: {question}\nAnswer:"
        )

    if strategy == "chain_of_thought":
        return (
            f"Context from a research paper:\n{context}\n\n"
            f"Question: {question}\n"
            "First, briefly reason step-by-step about which parts of the context are relevant "
            "(label this 'Reasoning:'). Then give a final concise answer (label this 'Answer:')."
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def llm_answer(question: str, sentences: list, strategy: str = "zero_shot", top_k: int = 5) -> str:
    """Retrieve relevant context, then generate an answer with the chosen prompt strategy."""
    context = " ".join(retrieve_relevant_sentences(question, sentences, top_k=top_k))
    if not context:
        context = " ".join(sentences[:top_k])  # fallback to first few sentences
    prompt = _build_prompt(strategy, question, context)
    return get_llm_response(prompt)


def compare_prompt_strategies(question: str, sentences: list) -> dict:
    """Run all three prompt strategies and return their answers side-by-side."""
    if not is_llm_configured():
        raise RuntimeError("No LLM configured - prompt strategy comparison requires an API key.")

    strategies = ["zero_shot", "few_shot", "chain_of_thought"]
    return {s: llm_answer(question, sentences, strategy=s) for s in strategies}


def answer_question(question: str, sentences: list, use_llm: bool, strategy: str = "zero_shot") -> dict:
    """Unified entry point used by the Streamlit app."""
    if use_llm and is_llm_configured():
        return {"answer": llm_answer(question, sentences, strategy=strategy), "method": f"LLM ({strategy})"}
    return {"answer": extractive_answer(question, sentences), "method": "extractive (TF-IDF retrieval)"}
