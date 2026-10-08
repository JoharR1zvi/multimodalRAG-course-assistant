# Index one lecture: read its chunks, turn them into vectors, and save them in Qdrant.
#
# Input : chunks.json            -> the chunks, each with the text to embed (embed_text)
# Output : points in the Qdrant collection of the current embedding model
#
# "Already indexed" means: the collection holds exactly as many points for this lecture
# as chunks.json has chunks. Then nothing is done, unless force=True.
#
# Usage (from the project root):
#   python -m src.database.indexing data/processed/lecture_01/chunks.json
#   python -m src.database.indexing data/processed/lecture_01/chunks.json --force

import json
import sys
from pathlib import Path

from src.database.vector_store import (
    count_points,
    create_collection_if_missing,
    delete_lecture,
    get_collection_name,
    open_client,
    upsert_chunks,
)
from src.embeddings.embedding_service import embed_texts, get_model_slug
from src.schemas.chunk import Chunk


def index_lecture(chunks_path: Path, *, force: bool = False, client=None) -> None:
    # client: pass an already open database (used by the tests). Normally it is opened here.

    with open(chunks_path, encoding="utf-8") as f:
        chunk_data = json.load(f)

    chunks = []
    for c in chunk_data:
        chunks.append(Chunk(**c))

    if len(chunks) == 0:
        print(f"No chunks in {chunks_path}, nothing to index.")
        return

    lecture_id = chunks[0].lecture_id
    collection_name = get_collection_name(get_model_slug())

    # Open the database ourselves unless one was given. Only one program can have it open at a time.
    opened_here = client is None
    if opened_here:
        client = open_client()

    try:
        already_stored = count_points(client, collection_name, lecture_id)

        # Same number of points as chunks: this lecture is already indexed
        if already_stored == len(chunks) and not force:
            print(f"{lecture_id} is already indexed ({already_stored} points in {collection_name}).")
            return

        # Embed the text of every chunk (vectors already computed before come from the disk cache)
        texts = []
        for chunk in chunks:
            texts.append(chunk.embed_text)

        vectors = embed_texts(texts)

        # The vector size is measured from the vectors themselves, never typed in
        vector_size = int(vectors.shape[1])
        create_collection_if_missing(client, collection_name, vector_size)

        # Remove the lecture's old points first, so chunks that no longer exist do not linger
        delete_lecture(client, collection_name, lecture_id)

        stored = upsert_chunks(client, collection_name, chunks, vectors)

        print(f"Indexed {stored} chunks of {lecture_id} into {collection_name}.")

    finally:
        if opened_here:
            client.close()


# Run the program
if __name__ == "__main__":

    chunks_path = Path(sys.argv[1])
    force = "--force" in sys.argv

    index_lecture(chunks_path, force=force)
