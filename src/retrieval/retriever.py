# Retrieval: from a question to the chunks most likely to contain the answer.
#
# The method is "dense" search (search by meaning):
#   1. turn the question into a vector, with the SAME model that embedded the chunks
#   2. ask Qdrant for the stored chunk vectors nearest to it
# With one signal that is all. With several signals (RETRIEVAL_SIGNALS, for example
# "speech,full": the speech alone and the full chunk text) step 2 is done once per signal and the
# lists are merged by reciprocal rank fusion (src/retrieval/fusion.py).
# A signal can also be a keyword search ("bm25", src/retrieval/keyword_search.py): it gives one more
# list for the same merge. Optionally (RERANK) a cross-encoder then re-sorts the best candidates
# (src/retrieval/reranker.py).

from src.config import FUSION_CANDIDATES, RERANK_CANDIDATES, RERANK_ENABLED, RETRIEVAL_SIGNALS
from src.database.vector_store import get_collection_name, load_all_chunks, open_client, search
from src.embeddings.embedding_service import embed_query, get_model_slug
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.keyword_search import dense_signals_only, is_keyword_signal, keyword_search
from src.retrieval.reranker import rerank_results


def format_timestamp(seconds: float) -> str:
    # 1415.7 -> "23:35"      5295 -> "1:28:15"
    # The same style a video player shows, so a student can jump straight to the moment.
    total_seconds = int(seconds)

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{remaining_seconds:02d}"

    return f"{minutes}:{remaining_seconds:02d}"


def retrieve(question: str, top_k: int = 5, lecture_id: str | None = None, client=None, signals: list | None = None, rerank: bool | None = None) -> list:
    # Returns a list of SearchResult, best match first.
    #   lecture_id : only search inside this lecture (None = all lectures)
    #   client     : an already open database (used by the tests). Normally it is opened here.
    #   signals    : which signals to search (default: RETRIEVAL_SIGNALS from the config)
    #   rerank     : True = a cross-encoder re-sorts the best candidates at the end
    #                (default: RERANK_ENABLED from the config)

    if signals is None:
        signals = RETRIEVAL_SIGNALS

    if rerank is None:
        rerank = RERANK_ENABLED

    # The first search fetches more than top_k when a reranker will pick the final top_k from them
    first_count = top_k
    if rerank:
        first_count = max(top_k, RERANK_CANDIDATES)

    # The question is only turned into a vector when at least one signal is a stored vector
    uses_vectors = len(dense_signals_only(signals)) > 0
    query_vector = None
    if uses_vectors:
        query_vector = embed_query(question)

    collection_name = get_collection_name(get_model_slug())

    # Open the database ourselves unless one was given. Only one program can have it open at a time.
    opened_here = client is None
    if opened_here:
        client = open_client()

    try:
        if len(signals) == 1 and not is_keyword_signal(signals[0]):
            # One vector signal: the plain search (the stored vectors are unnamed)
            results = search(client, collection_name, query_vector, top_k=first_count, lecture_id=lecture_id)
        else:
            # Several signals (or a keyword signal): get one list per signal, then merge the lists
            candidates = max(first_count, FUSION_CANDIDATES)

            # The keyword searches read the stored chunks; they are loaded once
            chunks = []
            if len(signals) > len(dense_signals_only(signals)):
                chunks = load_all_chunks(client, collection_name, lecture_id)

            result_lists = []
            for signal in signals:
                if is_keyword_signal(signal):
                    result_lists.append(keyword_search(chunks, question, signal, top_k=candidates))
                else:
                    result_lists.append(search(client, collection_name, query_vector, top_k=candidates, lecture_id=lecture_id, using=signal))

            if len(result_lists) == 1:
                results = result_lists[0][:first_count]
            else:
                results = reciprocal_rank_fusion(result_lists, first_count)
    finally:
        if opened_here:
            client.close()

    if rerank:
        results = rerank_results(question, results, top_k)
    else:
        results = results[:top_k]

    return results
