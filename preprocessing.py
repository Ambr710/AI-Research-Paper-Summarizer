"""
preprocessing.py
----------------
Lightweight NLP preprocessing that needs NO internet downloads
(no nltk punkt/wordnet downloads required) - everything is regex
and sklearn based so it works completely offline.
"""

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)

# Common noisy patterns found in academic PDFs (headers/footers/references)
_ARXIV_ID_RE = re.compile(r"arXiv:\d{4}\.\d{4,5}(v\d+)?", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"-\n")


def clean_text(raw_text: str) -> str:
    """Remove common PDF extraction artifacts (hyphenation, arXiv ids, extra whitespace)."""
    text = _HYPHEN_LINEBREAK_RE.sub("", raw_text)          # join hyphen-broken words
    text = _ARXIV_ID_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list:
    """
    Simple, dependency-free sentence splitter.
    Splits on '.', '!', '?' followed by whitespace + capital letter,
    while trying to avoid breaking on common abbreviations / decimals.
    """
    text = text.replace("\n", " ")
    # Protect common abbreviations from being treated as sentence boundaries
    protected = re.sub(r"\b(e\.g|i\.e|et al|Fig|Eq|Dr|vs)\.", r"\1<PERIOD>", text)
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", protected)
    sentences = [s.replace("<PERIOD>", ".").strip() for s in raw_sentences if s.strip()]
    # Filter out very short junk fragments (e.g. stray numbers, page headers)
    sentences = [s for s in sentences if len(s.split()) >= 4]
    return sentences


def tokenize_words(text: str) -> list:
    """Lowercase word tokenizer, strips punctuation."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


def extract_keywords(text: str, top_n: int = 15) -> list:
    """
    Extract top-N keywords using simple TF (term frequency) after
    stopword removal. Good enough for a 'basic' project and needs
    zero external downloads.
    """
    from collections import Counter

    tokens = remove_stopwords(tokenize_words(text))
    freq = Counter(tokens)
    return [word for word, _ in freq.most_common(top_n)]


def preprocess_pipeline(raw_text: str) -> dict:
    """Run the full preprocessing pipeline and return all intermediate artifacts."""
    cleaned = clean_text(raw_text)
    sentences = split_sentences(cleaned)
    keywords = extract_keywords(cleaned)
    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "keywords": keywords,
        "num_sentences": len(sentences),
        "num_words": len(tokenize_words(cleaned)),
    }
