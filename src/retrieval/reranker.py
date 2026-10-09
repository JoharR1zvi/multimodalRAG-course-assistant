# Reranking: a second, slower look at the best candidates of the first search.
#
# The first search (vectors + keywords) compares the question with each chunk from a distance:
# every chunk was turned into a fingerprint long before the question existed. A cross-encoder
# instead reads the question and one chunk TOGETHER, side by side, and gives one score for
# "does this chunk answer this question?". That is more accurate but much slower, so it only
# reads the few best candidates of the first search and puts them in a new order.
#
# The score of a reranked result is the model's relevance score between 0 and 1 (a sigmoid), not a
# cosine similarity. Unlike the cosine scores it may separate chunks that answer a question from
# chunks that only talk about the same topic; the evaluation prints how well it does.

from src.config import (
    FUSION_RRF_K,
    RERANK_BATCH_SIZE,
    RERANK_MAX_TOKENS,
    RERANK_MODE,
    RERANK_MODEL,
)
from src.schemas.chunk import SearchResult

_reranker = None


def load_reranker():
    # Loaded the first time it is needed (the first call also downloads the model, about 2 GB)
    global _reranker

    if _reranker is not None:
        return _reranker

    import torch
    from sentence_transformers import CrossEncoder

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Loading {RERANK_MODEL} on {device} (the first time, this also downloads it)...")
    model = CrossEncoder(RERANK_MODEL, device=device, max_length=RERANK_MAX_TOKENS)

    # Half precision on the GPU: half the memory, practically the same scores
    if device == "cuda":
        model.model.half()

    _reranker = model
    return _reranker


def rerank_results(question: str, results: list, top_k: int, mode: str | None = None) -> list:
    # results: the candidates from the first search (list of SearchResult, best first)
    # Returns the best top_k of them, best first. What the model reads for each chunk is its full
    # text (slide text, description and speech). Equal scores keep the order of the first search.
    #   mode "replace": sorted by the cross-encoder's score alone
    #   mode "blend"  : sorted by the first search's rank and the cross-encoder's rank together
    #                   (reciprocal rank fusion of the two orders)
    # Either way the score of a result is the cross-encoder's score.
    if mode is None:
        mode = RERANK_MODE

    if len(results) == 0:
        return []

    model = load_reranker()

    pairs = []
    for result in results:
        pairs.append((question, result.chunk.embed_text))

    scores = model.predict(pairs, batch_size=RERANK_BATCH_SIZE, show_progress_bar=False)

    order = sorted(range(len(results)), key=lambda i: (-float(scores[i]), i))

    if mode == "blend":
        # position 0 in `order` is the reranker's favourite; position i in `results` is the first search's rank - 1
        blended = {}
        for rerank_position in range(len(order)):
            i = order[rerank_position]
            first_rank = i + 1
            rerank_rank = rerank_position + 1
            blended[i] = 1.0 / (FUSION_RRF_K + first_rank) + 1.0 / (FUSION_RRF_K + rerank_rank)
        order = sorted(range(len(results)), key=lambda i: (-blended[i], i))

    reranked = []
    for i in order[:top_k]:
        reranked.append(SearchResult(chunk=results[i].chunk, score=float(scores[i]), method="rerank"))

    return reranked
