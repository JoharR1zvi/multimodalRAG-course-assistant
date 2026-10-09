# Merging several ranked lists of search results into one: reciprocal rank fusion.
#
# Each search ("signal") gives its own list, best first. A chunk that is high in several lists
# is probably a good answer, even if no list puts it first. Reciprocal rank fusion gives every
# chunk one point score for each list it appears in:
#
#     score = 1 / (k + rank)        (rank 1 is the best place; k = 60 is the usual choice)
#
# and adds the scores up. A chunk at rank 1 in one list and rank 3 in another gets
# 1/61 + 1/63. Only the ranks count, never the similarity numbers, so two lists whose scores
# mean different things (two texts, or later a keyword search) can be merged fairly.

from src.schemas.chunk import SearchResult

DEFAULT_K = 60


def reciprocal_rank_fusion(result_lists: list, top_k: int, k: int = DEFAULT_K) -> list:
    # result_lists: a list of lists of SearchResult, each list best first
    # Returns the best top_k merged results, best first. Each one keeps the chunk, and its score
    # is the highest similarity the chunk had in any DENSE list (so it still reads like a
    # similarity); the method says "fusion". Keyword (BM25) scores are not similarities, so they
    # never become the shown score; a chunk found only by the keyword search shows 0.0.
    # Ties keep the order in which the chunks were first seen.

    fused_scores = {}        # chunk id -> sum of 1 / (k + rank)
    best_similarity = {}     # chunk id -> highest dense similarity in any list (None = none yet)
    chunk_of = {}            # chunk id -> the Chunk
    first_seen = {}          # chunk id -> order of first appearance (for ties)

    for result_list in result_lists:
        for position in range(len(result_list)):
            result = result_list[position]
            chunk_id = result.chunk.chunk_id
            rank = position + 1

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
                best_similarity[chunk_id] = None
                chunk_of[chunk_id] = result.chunk
                first_seen[chunk_id] = len(first_seen)

            fused_scores[chunk_id] = fused_scores[chunk_id] + 1.0 / (k + rank)

            if result.method != "bm25":
                if best_similarity[chunk_id] is None or result.score > best_similarity[chunk_id]:
                    best_similarity[chunk_id] = result.score

    # Highest fused score first; the earlier-seen chunk wins a tie
    chunk_ids = list(fused_scores.keys())
    chunk_ids.sort(key=lambda chunk_id: (-fused_scores[chunk_id], first_seen[chunk_id]))

    merged = []
    for chunk_id in chunk_ids[:top_k]:
        shown_score = best_similarity[chunk_id]
        if shown_score is None:
            shown_score = 0.0
        merged.append(SearchResult(chunk=chunk_of[chunk_id], score=shown_score, method="fusion"))

    return merged
