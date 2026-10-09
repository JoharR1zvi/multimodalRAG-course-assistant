# Keyword search (BM25): find the chunks that contain the words of the question.
#
# The dense search looks for chunks with a similar MEANING. This search looks for the same WORDS,
# like Ctrl+F, but it ranks the chunks instead of just finding them. BM25 gives a chunk points for
# every question word it contains, with two rules:
#   - a rare word counts more than a common one ("bootstrap" is worth more than "the")
#   - repeating a word helps, but less and less (ten mentions are not ten times better than one)
# It also takes the length of the chunk into account, so a long chunk does not win just by
# having more words.
#
# A keyword search is a "signal" like the stored vectors are (see RETRIEVAL_SIGNALS):
#   "bm25"        searches the full chunk text (slide text, description and speech)
#   "bm25_speech" searches only what was said
# Its list is merged with the other lists by reciprocal rank fusion (fusion.py).
# Nothing extra is stored: the index is built from the chunks that are in the database, and
# it is small (96 chunks take a few milliseconds).

import re

from rank_bm25 import BM25Okapi

from src.schemas.chunk import SearchResult

# The keyword signals, and which field of a chunk each one reads
KEYWORD_SIGNALS = {
    "bm25": "embed_text",
    "bm25_speech": "text",
}

# Words that carry no topic. Left out of the question and of the chunks, so that they cannot
# give a chunk points just because it is long.
STOP_WORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "from", "by", "with", "and", "or", "but",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "may", "might", "this", "that", "these", "those",
    "it", "its", "i", "you", "he", "she", "we", "they", "me", "my", "your", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how", "not", "no", "yes", "as", "if",
    "so", "than", "then", "there", "here", "about", "into", "over", "also",
}


def is_keyword_signal(signal: str) -> bool:
    return signal in KEYWORD_SIGNALS


def dense_signals_only(signals: list) -> list:
    # The signals that are stored vectors (everything that is not a keyword signal)
    return [signal for signal in signals if not is_keyword_signal(signal)]


def tokenize(text: str) -> list:
    # "What is the F1-score?" -> ["f1", "score"]
    # Lower case, cut into pieces made of letters and digits, drop the stop words.
    words = re.findall(r"[a-z0-9]+", text.lower())

    kept = []
    for word in words:
        if word not in STOP_WORDS:
            kept.append(word)

    return kept


def keyword_search(chunks: list, question: str, signal: str = "bm25", top_k: int = 10) -> list:
    # chunks: the Chunk objects to search (all of them, or one lecture's)
    # Returns a list of SearchResult, best first. Chunks that share no word with the question
    # are left out. The score is the BM25 score: it is NOT a similarity between 0 and 1, and it
    # cannot be compared with the dense scores (the fusion only uses the ranks).
    if signal not in KEYWORD_SIGNALS:
        raise ValueError(f"Unknown keyword signal {signal!r}. Use one of {list(KEYWORD_SIGNALS)}.")

    if len(chunks) == 0:
        return []

    field = KEYWORD_SIGNALS[signal]

    chunk_words = []
    for chunk in chunks:
        chunk_words.append(tokenize(getattr(chunk, field)))

    question_words = tokenize(question)
    if len(question_words) == 0:
        return []

    index = BM25Okapi(chunk_words)
    scores = index.get_scores(question_words)

    # Best score first; the earlier chunk wins a tie
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))

    results = []
    for i in order[:top_k]:
        if scores[i] <= 0:
            break
        results.append(SearchResult(chunk=chunks[i], score=float(scores[i]), method="bm25"))

    return results
