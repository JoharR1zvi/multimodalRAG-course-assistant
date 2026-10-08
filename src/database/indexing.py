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

from src.config import RETRIEVAL_SIGNALS
from src.database.vector_store import (
    count_points,
    create_collection_if_missing,
    delete_lecture,
    get_collection_name,
    get_vector_names,
    open_client,
    upsert_chunks,
)
from src.embeddings.embedding_service import embed_texts, get_model_slug
from src.schemas.chunk import Chunk

# The signals a search can use, and which text of a chunk each one embeds
SIGNAL_NAMES = ["full", "speech"]


def texts_for_signal(chunks: list, signal: str) -> list:
    # The text that is embedded for one signal, one per chunk, in order.
    #   "full"   : embed_text (slide text, description and speech, as chunking decided)
    #   "speech" : only what was said
    if signal not in SIGNAL_NAMES:
        raise ValueError(f"Unknown retrieval signal {signal!r}. Use one of {SIGNAL_NAMES}.")

    texts = []
    for chunk in chunks:
        if signal == "speech":
            texts.append(chunk.text)
        else:
            texts.append(chunk.embed_text)

    return texts


def index_lecture(chunks_path: Path, *, force: bool = False, client=None, signals: list | None = None) -> None:
    # client : pass an already open database (used by the tests). Normally it is opened here.
    # signals: which texts to embed (default: RETRIEVAL_SIGNALS from the config). One signal gives
    #          one unnamed vector per chunk; several give one named vector per signal.
    if signals is None:
        signals = RETRIEVAL_SIGNALS

    if len(signals) == 0:
        raise ValueError("RETRIEVAL_SIGNALS is empty")

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
        # A database keeps ONE layout of vectors. Refuse to mix, with a hint how to fix it.
        if len(signals) > 1:
            expected_names = sorted(signals)
        else:
            expected_names = None

        if client.collection_exists(collection_name):
            existing_names = get_vector_names(client, collection_name)
            if existing_names != expected_names:
                raise ValueError(
                    f"{collection_name} was built with the vectors {existing_names}, but RETRIEVAL_SIGNALS asks for "
                    f"{expected_names}. Use another QDRANT_PATH, or delete the old database folder."
                )

        already_stored = count_points(client, collection_name, lecture_id)

        # Same number of points as chunks: this lecture is already indexed
        if already_stored == len(chunks) and not force:
            print(f"{lecture_id} is already indexed ({already_stored} points in {collection_name}).")
            return

        # Embed the text of every chunk, once per signal (vectors computed before come from the disk cache)
        vectors_by_signal = {}
        for signal in signals:
            vectors_by_signal[signal] = embed_texts(texts_for_signal(chunks, signal))

        # The vector size is measured from the vectors themselves, never typed in
        vector_size = int(vectors_by_signal[signals[0]].shape[1])
        create_collection_if_missing(client, collection_name, vector_size, vector_names=expected_names)

        if len(signals) > 1:
            vectors = vectors_by_signal
        else:
            vectors = vectors_by_signal[signals[0]]

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
