# Tests for reciprocal rank fusion (src/retrieval/fusion.py), with small hand-made lists.

import pytest

from src.retrieval.fusion import reciprocal_rank_fusion
from src.schemas.chunk import Chunk, SearchResult


def make_result(chunk_id, score=0.5):
    chunk = Chunk(
        chunk_id=chunk_id,
        course_id="course_test",
        lecture_id="lecture_a",
        start_timestamp=0.0,
        end_timestamp=100.0,
        slide_timestamps=[0.0],
        image_paths=["frame_0.jpg"],
        slide_titles=["A title"],
        text="speech",
        slide_text="",
        cleaned_text="",
        clean_text="",
        slide_description="",
        embed_text="embedded",
    )
    return SearchResult(chunk=chunk, score=score, method="dense")


def ids(results):
    return [result.chunk.chunk_id for result in results]


def test_a_chunk_that_is_in_both_lists_beats_chunks_that_are_in_one():
    list_one = [make_result("x"), make_result("y"), make_result("z")]
    list_two = [make_result("y"), make_result("w")]

    merged = reciprocal_rank_fusion([list_one, list_two], top_k=10)

    # y: 1/62 + 1/61, x: 1/61, w: 1/62, z: 1/63
    assert ids(merged) == ["y", "x", "w", "z"]


def test_a_keyword_score_is_never_shown_as_a_similarity():
    # BM25 scores are not between 0 and 1. The shown score comes from the dense lists only;
    # a chunk found only by the keyword search shows 0.0.
    dense_list = [make_result("x", score=0.8)]
    keyword_list = [
        SearchResult(chunk=make_result("x").chunk, score=14.2, method="bm25"),
        SearchResult(chunk=make_result("y").chunk, score=9.0, method="bm25"),
    ]

    merged = reciprocal_rank_fusion([dense_list, keyword_list], top_k=10)

    assert ids(merged) == ["x", "y"]
    assert merged[0].score == 0.8
    assert merged[1].score == 0.0


def test_only_the_ranks_count_not_the_similarity_numbers():
    # The first list has very high scores and the second very low ones: it makes no difference
    list_one = [make_result("a", score=0.99), make_result("b", score=0.98)]
    list_two = [make_result("b", score=0.01), make_result("a", score=0.005)]

    merged = reciprocal_rank_fusion([list_one, list_two], top_k=10)

    # a: 1/61 + 1/62, b: 1/62 + 1/61: a tie, and "a" was seen first
    assert ids(merged) == ["a", "b"]


def test_the_score_of_a_merged_result_is_its_best_similarity_and_the_method_says_fusion():
    list_one = [make_result("a", score=0.4)]
    list_two = [make_result("a", score=0.7)]

    merged = reciprocal_rank_fusion([list_one, list_two], top_k=5)

    assert len(merged) == 1
    assert merged[0].score == pytest.approx(0.7)
    assert merged[0].method == "fusion"


def test_top_k_limits_the_merged_list():
    list_one = [make_result("a"), make_result("b"), make_result("c")]
    list_two = [make_result("d"), make_result("e")]

    merged = reciprocal_rank_fusion([list_one, list_two], top_k=2)

    assert len(merged) == 2


def test_a_tie_keeps_the_chunk_that_was_seen_first():
    list_one = [make_result("first")]
    list_two = [make_result("second")]

    merged = reciprocal_rank_fusion([list_one, list_two], top_k=5)

    assert ids(merged) == ["first", "second"]


def test_empty_lists_give_an_empty_result():
    assert reciprocal_rank_fusion([], top_k=5) == []
    assert reciprocal_rank_fusion([[], []], top_k=5) == []


def test_the_constant_k_changes_how_much_the_top_rank_is_worth():
    # "top" is first in one list only; "both" is second in both lists.
    list_one = [make_result("top"), make_result("both")]
    list_two = [make_result("other"), make_result("both")]

    # With k = 1: top = 1/2 = 0.5, both = 1/3 + 1/3 = 0.667, so appearing in both lists wins
    assert ids(reciprocal_rank_fusion([list_one, list_two], top_k=3, k=1))[0] == "both"

    # With k = 60 the same ordering holds, because 1/62 + 1/62 = 0.0323 beats 1/61 = 0.0164
    assert ids(reciprocal_rank_fusion([list_one, list_two], top_k=3, k=60))[0] == "both"
