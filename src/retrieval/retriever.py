# Retrieval: from a question to the chunks most likely to contain the answer.
#
# For now there is one method, "dense" search (search by meaning):
#   1. turn the question into a vector, with the SAME model that embedded the chunks
#   2. ask Qdrant for the stored chunk vectors nearest to it
# Keyword search (BM25) and a re-ranking step come later and plug in here.

from src.database.vector_store import get_collection_name, open_client, search
from src.embeddings.embedding_service import embed_query, get_model_slug


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


def retrieve(question: str, top_k: int = 5, lecture_id: str | None = None, client=None) -> list:
    # Returns a list of SearchResult, best match first.
    #   lecture_id : only search inside this lecture (None = all lectures)
    #   client     : an already open database (used by the tests). Normally it is opened here.

    query_vector = embed_query(question)
    collection_name = get_collection_name(get_model_slug())

    # Open the database ourselves unless one was given. Only one program can have it open at a time.
    opened_here = client is None
    if opened_here:
        client = open_client()

    try:
        results = search(client, collection_name, query_vector, top_k=top_k, lecture_id=lecture_id)
    finally:
        if opened_here:
            client.close()

    return results
