# Tests for vector_store.py and indexing.py.
#
# Qdrant has an in-memory mode (a throwaway database in RAM), so these tests need no folder,
# no server and no embedding model. Vectors are tiny (3 numbers) and chosen by hand so that
# we know which chunk is "closest" to which question.

import json

import numpy as np
import pytest

from src.database import indexing
from src.database.indexing import index_lecture
from src.database.vector_store import (
    count_points,
    create_collection_if_missing,
    delete_lecture,
    get_collection_name,
    make_point_id,
    open_client,
    search,
    upsert_chunks,
)
from src.schemas.chunk import Chunk

COLLECTION = "test_collection"


def make_chunk(chunk_id, lecture_id="lecture_a", start=0.0, title="Some title"):
    return Chunk(
        chunk_id=chunk_id,
        course_id="course_test",
        lecture_id=lecture_id,
        start_timestamp=start,
        end_timestamp=start + 100.0,
        slide_timestamps=[start],
        image_paths=[f"frame_{start}.jpg"],
        slide_titles=[title],
        text="some speech",
        slide_text="raw",
        cleaned_text="cleaned",
        clean_text="vision",
        slide_description="",
        embed_text=f"embed text of {chunk_id}",
    )


@pytest.fixture
def client():
    # a fresh, empty, in-memory database for every test
    database = open_client(":memory:")
    create_collection_if_missing(database, COLLECTION, 3)
    yield database
    database.close()


def test_the_same_chunk_id_always_gives_the_same_point_id():
    assert make_point_id("lecture_a_10") == make_point_id("lecture_a_10")
    assert make_point_id("lecture_a_10") != make_point_id("lecture_a_11")


def test_the_collection_is_named_after_the_model():
    assert get_collection_name("bge-m3") == "course_chunks__bge-m3"
    assert get_collection_name("bge-m3") != get_collection_name("other-model")


def test_saving_the_same_chunks_twice_does_not_create_duplicates(client):
    chunks = [make_chunk("lecture_a_0"), make_chunk("lecture_a_100", start=100.0)]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    upsert_chunks(client, COLLECTION, chunks, vectors)
    upsert_chunks(client, COLLECTION, chunks, vectors)

    assert count_points(client, COLLECTION) == 2


def test_search_returns_the_nearest_chunk_first_with_its_citation_data(client):
    chunks = [
        make_chunk("lecture_a_0", start=0.0, title="About cats"),
        make_chunk("lecture_a_100", start=100.0, title="About dogs"),
        make_chunk("lecture_a_200", start=200.0, title="About fish"),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    upsert_chunks(client, COLLECTION, chunks, vectors)

    # a question that points almost exactly at the "dogs" vector
    results = search(client, COLLECTION, [0.1, 0.9, 0.0], top_k=3)

    assert len(results) == 3
    assert results[0].chunk.chunk_id == "lecture_a_100"
    assert results[0].chunk.slide_titles == ["About dogs"]
    assert results[0].chunk.start_timestamp == 100.0
    assert results[0].method == "dense"

    # best first: scores never go up as we move down the list
    assert results[0].score >= results[1].score >= results[2].score


def test_top_k_limits_the_number_of_results(client):
    chunks = [make_chunk("lecture_a_0"), make_chunk("lecture_a_100", start=100.0), make_chunk("lecture_a_200", start=200.0)]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    upsert_chunks(client, COLLECTION, chunks, vectors)

    assert len(search(client, COLLECTION, [1.0, 0.0, 0.0], top_k=2)) == 2


def test_search_can_be_limited_to_one_lecture(client):
    chunks = [
        make_chunk("lecture_a_0", lecture_id="lecture_a"),
        make_chunk("lecture_b_0", lecture_id="lecture_b"),
    ]
    # the question is closest to lecture_a's chunk
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    upsert_chunks(client, COLLECTION, chunks, vectors)

    only_b = search(client, COLLECTION, [1.0, 0.0, 0.0], top_k=5, lecture_id="lecture_b")

    assert len(only_b) == 1
    assert only_b[0].chunk.lecture_id == "lecture_b"


def test_searching_a_collection_that_does_not_exist_gives_no_results(client):
    assert search(client, "no_such_collection", [1.0, 0.0, 0.0]) == []
    assert count_points(client, "no_such_collection") == 0


def test_a_different_number_of_chunks_and_vectors_is_refused(client):
    with pytest.raises(ValueError):
        upsert_chunks(client, COLLECTION, [make_chunk("lecture_a_0")], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_deleting_a_lecture_leaves_the_other_lectures_alone(client):
    chunks = [
        make_chunk("lecture_a_0", lecture_id="lecture_a"),
        make_chunk("lecture_b_0", lecture_id="lecture_b"),
    ]
    upsert_chunks(client, COLLECTION, chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    delete_lecture(client, COLLECTION, "lecture_a")

    assert count_points(client, COLLECTION, "lecture_a") == 0
    assert count_points(client, COLLECTION, "lecture_b") == 1


# ---------- indexing a whole lecture ----------

class FakeEmbedder:
    # Replaces the real embedding model: counts how often it is used and gives each text
    # a different vector.
    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        self.calls = self.calls + 1
        vectors = []
        for i in range(len(texts)):
            vectors.append([1.0, float(i), 0.0])
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def fake_embedder(monkeypatch):
    fake = FakeEmbedder()
    monkeypatch.setattr(indexing, "embed_texts", fake)
    monkeypatch.setattr(indexing, "get_model_slug", lambda: "fake-model")
    return fake


def write_chunks_file(path, chunks):
    data = []
    for chunk in chunks:
        data.append(chunk.model_dump())
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_indexing_a_lecture_stores_one_point_per_chunk(tmp_path, fake_embedder):
    chunks_path = tmp_path / "chunks.json"
    write_chunks_file(chunks_path, [make_chunk("lecture_a_0"), make_chunk("lecture_a_100", start=100.0)])
    database = open_client(":memory:")

    index_lecture(chunks_path, client=database)

    collection = get_collection_name("fake-model")
    assert count_points(database, collection, "lecture_a") == 2


def test_indexing_again_does_nothing_unless_forced(tmp_path, fake_embedder):
    chunks_path = tmp_path / "chunks.json"
    write_chunks_file(chunks_path, [make_chunk("lecture_a_0"), make_chunk("lecture_a_100", start=100.0)])
    database = open_client(":memory:")

    index_lecture(chunks_path, client=database)
    assert fake_embedder.calls == 1

    index_lecture(chunks_path, client=database)
    assert fake_embedder.calls == 1            # already indexed: the embedder was not used again

    index_lecture(chunks_path, client=database, force=True)
    assert fake_embedder.calls == 2            # forced: done again


def test_reindexing_after_the_chunks_changed_removes_the_old_points(tmp_path, fake_embedder):
    chunks_path = tmp_path / "chunks.json"
    database = open_client(":memory:")
    collection = get_collection_name("fake-model")

    # first version: 3 chunks
    write_chunks_file(chunks_path, [make_chunk("lecture_a_0"), make_chunk("lecture_a_100", start=100.0), make_chunk("lecture_a_200", start=200.0)])
    index_lecture(chunks_path, client=database)
    assert count_points(database, collection, "lecture_a") == 3

    # the lecture was chunked again and now has 2 different chunks: the 3 old points must go
    write_chunks_file(chunks_path, [make_chunk("lecture_a_0"), make_chunk("lecture_a_150", start=150.0)])
    index_lecture(chunks_path, client=database)

    assert count_points(database, collection, "lecture_a") == 2


def test_indexing_one_lecture_does_not_touch_another(tmp_path, fake_embedder):
    database = open_client(":memory:")
    collection = get_collection_name("fake-model")

    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    write_chunks_file(first, [make_chunk("lecture_a_0", lecture_id="lecture_a")])
    write_chunks_file(second, [make_chunk("lecture_b_0", lecture_id="lecture_b")])

    index_lecture(first, client=database)
    index_lecture(second, client=database)

    assert count_points(database, collection, "lecture_a") == 1
    assert count_points(database, collection, "lecture_b") == 1
    assert count_points(database, collection) == 2
