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
    # is the highest similarity the chunk had in any list (so it still reads like a similarity);
    # the method says "fusion". Ties keep the order in which the chunks were first seen.

    fused_scores = {}        # chunk id -> sum of 1 / (k + rank)
    best_similarity = {}     # chunk id -> highest similarity score in any list
    chunk_of = {}            # chunk id -> the Chunk
    first_seen = {}          # chunk id -> order of first appearance (for ties)

    for result_list in result_lists:
        for position in range(len(result_list)):
            result = result_list[position]
            chunk_id = result.chunk.chunk_id
            rank = position + 1

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0
                best_similarity[chunk_id] = result.score
                chunk_of[chunk_id] = result.chunk
                first_seen[chunk_id] = len(first_seen)

            fused_scores[chunk_id] = fused_scores[chunk_id] + 1.0 / (k + rank)

            if result.score > best_similarity[chunk_id]:
                best_similarity[chunk_id] = result.score

    # Highest fused score first; the earlier-seen chunk wins a tie
    chunk_ids = list(fused_scores.keys())
    chunk_ids.sort(key=lambda chunk_id: (-fused_scores[chunk_id], first_seen[chunk_id]))

    merged = []
    for chunk_id in chunk_ids[:top_k]:
        merged.append(SearchResult(chunk=chunk_of[chunk_id], score=best_similarity[chunk_id], method="fusion"))

    return merged
