# The vector store: a small wrapper around Qdrant, the database that finds the stored vectors
# closest to a question's vector.
#
# Vocabulary:
#   collection : like a table. All chunks embedded with ONE model live in one collection.
#   point      : like a row. One chunk = one point = a vector + a payload.
#   payload    : the chunk's normal information (lecture, times, slide titles, text...),
#                stored next to the vector, so a search result carries its own citation.
#
# Qdrant runs inside our own program ("embedded mode") and keeps its files in a folder,
# so there is no server to install or start.
#
# This file knows nothing about embedding models: it receives vectors and gives them back.

import uuid

from qdrant_client import QdrantClient, models

from src.config import QDRANT_PATH, QDRANT_COLLECTION_PREFIX
from src.schemas.chunk import Chunk, SearchResult


def get_collection_name(model_slug: str) -> str:
    # "bge-m3" -> "course_chunks__bge-m3". One collection per embedding model.
    return f"{QDRANT_COLLECTION_PREFIX}__{model_slug}"


def open_client(path: str = QDRANT_PATH) -> QdrantClient:
    # path=":memory:" gives a throwaway database that lives only in RAM (used by the tests)
    if path == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(path=path)


def make_point_id(chunk_id: str) -> str:
    # Qdrant wants a number or a UUID as a point id, not free text like "lecture_01_1074".
    # uuid5 turns the same text into the same UUID every time, so saving a chunk twice
    # overwrites its point instead of creating a second one.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def to_float_list(vector) -> list:
    # Qdrant wants a plain list of Python floats (our vectors are numpy arrays)
    numbers = []
    for number in vector:
        numbers.append(float(number))
    return numbers


def create_collection_if_missing(client: QdrantClient, collection_name: str, vector_size: int, vector_names: list | None = None) -> None:
    # vector_names: None = every point has one vector (the usual case).
    # A list of names, such as ["speech", "full"] = every point has one vector PER NAME, so one
    # chunk can be searched by its speech and by its full text with the same stored payload.
    if client.collection_exists(collection_name):
        return

    # cosine = compare the DIRECTION of two vectors, which is what "same meaning" is
    one_vector = models.VectorParams(size=vector_size, distance=models.Distance.COSINE)

    if vector_names is None:
        vectors_config = one_vector
    else:
        vectors_config = {}
        for name in vector_names:
            vectors_config[name] = one_vector

    client.create_collection(collection_name=collection_name, vectors_config=vectors_config)


def get_vector_names(client: QdrantClient, collection_name: str) -> list | None:
    # The names of the vectors of an existing collection, or None when its points have one
    # unnamed vector. Returns None as well when the collection does not exist.
    if not client.collection_exists(collection_name):
        return None

    vectors_config = client.get_collection(collection_name).config.params.vectors

    if isinstance(vectors_config, dict):
        return sorted(vectors_config.keys())

    return None


def make_lecture_filter(lecture_id: str) -> models.Filter:
    # "only points whose payload says lecture_id = ..."
    return models.Filter(
        must=[models.FieldCondition(key="lecture_id", match=models.MatchValue(value=lecture_id))]
    )


def count_points(client: QdrantClient, collection_name: str, lecture_id: str | None = None) -> int:
    # How many points are stored (in total, or for one lecture). 0 if the collection does not exist yet.
    if not client.collection_exists(collection_name):
        return 0

    if lecture_id is None:
        return client.count(collection_name=collection_name, exact=True).count

    return client.count(
        collection_name=collection_name,
        count_filter=make_lecture_filter(lecture_id),
        exact=True,
    ).count


def delete_lecture(client: QdrantClient, collection_name: str, lecture_id: str) -> None:
    # Removes every point of one lecture (used before re-indexing it)
    if not client.collection_exists(collection_name):
        return

    client.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(filter=make_lecture_filter(lecture_id)),
    )


def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list, vectors) -> int:
    # Saves chunks with their vectors. "Upsert" = insert, or overwrite if the id already exists,
    # so running it twice never duplicates anything.
    # chunks : list of Chunk
    # vectors: one vector per chunk, in the same order (one unnamed vector per chunk), OR a dict
    #          {name: one vector per chunk} when the collection has named vectors

    if isinstance(vectors, dict):
        vectors_by_name = vectors
    else:
        vectors_by_name = None

    if vectors_by_name is None:
        vector_lists = [vectors]
    else:
        vector_lists = list(vectors_by_name.values())

    for vector_list in vector_lists:
        if len(chunks) != len(vector_list):
            raise ValueError(f"{len(chunks)} chunks but {len(vector_list)} vectors")

    points = []

    for i in range(len(chunks)):
        chunk = chunks[i]

        if vectors_by_name is None:
            point_vector = to_float_list(vectors[i])
        else:
            point_vector = {}
            for name in vectors_by_name:
                point_vector[name] = to_float_list(vectors_by_name[name][i])

        point = models.PointStruct(
            id=make_point_id(chunk.chunk_id),
            vector=point_vector,
            payload=chunk.model_dump(),
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points)

    return len(points)


def load_all_chunks(client: QdrantClient, collection_name: str, lecture_id: str | None = None) -> list:
    # Every stored chunk (or those of one lecture), read from the payloads, in time order.
    # The keyword search builds its index from these, so it searches exactly what is stored.
    if not client.collection_exists(collection_name):
        return []

    scroll_filter = None
    if lecture_id is not None:
        scroll_filter = make_lecture_filter(lecture_id)

    chunks = []
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            chunks.append(Chunk(**point.payload))

        if next_offset is None:
            break

    chunks.sort(key=lambda chunk: (chunk.lecture_id, chunk.start_timestamp))

    return chunks


def search(
    client: QdrantClient,
    collection_name: str,
    query_vector,
    top_k: int = 5,
    lecture_id: str | None = None,
    using: str | None = None,
) -> list:
    # The nearest chunks to a question's vector, best first. Optionally only inside one lecture.
    # using: the name of the vector to search, when the collection has named vectors.
    # Returns a list of SearchResult (chunk + score + method).

    if not client.collection_exists(collection_name):
        return []

    query_filter = None
    if lecture_id is not None:
        query_filter = make_lecture_filter(lecture_id)

    response = client.query_points(
        collection_name=collection_name,
        query=to_float_list(query_vector),
        using=using,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )

    results = []

    for point in response.points:
        result = SearchResult(
            chunk=Chunk(**point.payload),
            score=point.score,
            method="dense",
        )
        results.append(result)

    return results
