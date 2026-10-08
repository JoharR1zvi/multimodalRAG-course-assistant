# Tests for the retriever and the search command's text output.
#
# Same trick as the vector store tests: an in-memory Qdrant with tiny hand-made vectors,
# and a fake "embed_query" so the real model is never loaded.

import numpy as np
import pytest

from src.database.vector_store import (
    create_collection_if_missing,
    get_collection_name,
    open_client,
    upsert_chunks,
)
from src.retrieval import retriever
from src.retrieval.retriever import format_timestamp, retrieve
from src.schemas.chunk import Chunk, SearchResult
from src.search import PREVIEW_CHARACTERS, format_result, unique_titles


def make_chunk(chunk_id, lecture_id="lecture_a", start=0.0, end=100.0, titles=None, text="some speech"):
    if titles is None:
        titles = ["A title"]
    return Chunk(
        chunk_id=chunk_id,
        course_id="course_test",
        lecture_id=lecture_id,
        start_timestamp=start,
        end_timestamp=end,
        slide_timestamps=[start],
        image_paths=[f"frame_{start}.jpg"],
        slide_titles=titles,
        text=text,
        slide_text="",
        cleaned_text="",
        clean_text="",
        slide_description="",
        embed_text="embedded",
    )


@pytest.fixture
def database(monkeypatch):
    # An in-memory database holding three chunks, and a fake question embedder that
    # always points at the "dogs" chunk.
    monkeypatch.setattr(retriever, "get_model_slug", lambda: "fake-model")
    monkeypatch.setattr(retriever, "embed_query", lambda question: np.array([0.0, 1.0, 0.0]))

    client = open_client(":memory:")
    collection = get_collection_name("fake-model")
    create_collection_if_missing(client, collection, 3)

    chunks = [
        make_chunk("lecture_a_0", lecture_id="lecture_a", start=0.0, titles=["About cats"]),
        make_chunk("lecture_a_100", lecture_id="lecture_a", start=100.0, titles=["About dogs"]),
        make_chunk("lecture_b_0", lecture_id="lecture_b", start=0.0, titles=["More dogs"]),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.9, 0.0]]
    upsert_chunks(client, collection, chunks, vectors)

    yield client
    client.close()


def test_the_best_matching_chunk_comes_first(database):
    results = retrieve("anything", top_k=3, client=database)

    assert results[0].chunk.chunk_id == "lecture_a_100"
    assert results[1].chunk.chunk_id == "lecture_b_0"
    assert results[2].chunk.chunk_id == "lecture_a_0"


def test_top_k_limits_the_results(database):
    assert len(retrieve("anything", top_k=2, client=database)) == 2


def test_the_search_can_be_limited_to_one_lecture(database):
    results = retrieve("anything", top_k=5, lecture_id="lecture_b", client=database)

    assert len(results) == 1
    assert results[0].chunk.lecture_id == "lecture_b"


def test_two_signals_are_searched_separately_and_merged_by_rank(monkeypatch):
    # Three chunks, two stored vectors each. For the question [1, 0, 0]:
    #   by speech the order is  A, B, C      by full text the order is  B, C, A
    # Fused: B is high in both lists and wins, then A, then C.
    monkeypatch.setattr(retriever, "get_model_slug", lambda: "fake-model")
    monkeypatch.setattr(retriever, "embed_query", lambda question: np.array([1.0, 0.0, 0.0]))

    client = open_client(":memory:")
    collection = get_collection_name("fake-model")
    create_collection_if_missing(client, collection, 3, vector_names=["speech", "full"])

    chunks = [make_chunk("A", start=0.0), make_chunk("B", start=100.0), make_chunk("C", start=200.0)]
    vectors = {
        "speech": [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.6, 0.8, 0.0]],
        "full": [[0.6, 0.8, 0.0], [1.0, 0.0, 0.0], [0.8, 0.6, 0.0]],
    }
    upsert_chunks(client, collection, chunks, vectors)

    fused = retrieve("anything", top_k=3, client=client, signals=["speech", "full"])

    assert [result.chunk.chunk_id for result in fused] == ["B", "A", "C"]
    assert fused[0].method == "fusion"
    client.close()


def test_one_signal_is_the_plain_search_with_the_dense_method(database):
    results = retrieve("anything", top_k=3, client=database, signals=["full"])

    assert results[0].method == "dense"


def test_an_empty_database_gives_no_results(monkeypatch):
    monkeypatch.setattr(retriever, "get_model_slug", lambda: "fake-model")
    monkeypatch.setattr(retriever, "embed_query", lambda question: np.array([0.0, 1.0, 0.0]))

    assert retrieve("anything", client=open_client(":memory:")) == []


def test_timestamps_are_shown_like_a_video_player():
    assert format_timestamp(0) == "0:00"
    assert format_timestamp(65) == "1:05"
    assert format_timestamp(1415.7) == "23:35"
    assert format_timestamp(3600) == "1:00:00"
    assert format_timestamp(5295) == "1:28:15"


def test_repeated_titles_are_shown_once_in_order():
    assert unique_titles(["B", "B", "A", "B", "C"]) == ["B", "A", "C"]
    assert unique_titles([]) == []


def test_a_result_is_shown_with_score_lecture_times_titles_and_speech():
    chunk = make_chunk("lecture_a_1074", start=1074.8, end=1245.9, titles=["Same", "Same", "Other"], text="the speech of the chunk")
    result = SearchResult(chunk=chunk, score=0.6034, method="dense")

    text = format_result(1, result)

    assert "1. score 0.603" in text
    assert "lecture_a" in text
    assert "17:54 - 20:45" in text
    assert "Slides: Same | Other" in text
    assert "the speech of the chunk" in text


def test_a_long_speech_is_cut_at_a_word_boundary_with_dots():
    long_speech = "word " * 200
    chunk = make_chunk("lecture_a_0", text=long_speech)
    result = SearchResult(chunk=chunk, score=0.5, method="dense")

    text = format_result(1, result)
    speech_line = text.split("\n")[2]

    assert speech_line.endswith(' ..."')
    assert len(speech_line) < PREVIEW_CHARACTERS + 20


def test_a_chunk_without_titles_says_so():
    chunk = make_chunk("lecture_a_0", titles=[])
    result = SearchResult(chunk=chunk, score=0.5, method="dense")

    assert "(no slide titles)" in format_result(1, result)
