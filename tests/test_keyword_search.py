# Tests for the keyword search (BM25) in src/retrieval/keyword_search.py, with tiny hand-made chunks.

import pytest

from src.retrieval.keyword_search import dense_signals_only, is_keyword_signal, keyword_search, tokenize
from src.schemas.chunk import Chunk


def make_chunk(chunk_id, speech="", full="", start=0.0):
    return Chunk(
        chunk_id=chunk_id,
        course_id="course_test",
        lecture_id="lecture_a",
        start_timestamp=start,
        end_timestamp=start + 100.0,
        slide_timestamps=[start],
        image_paths=[f"frame_{start}.jpg"],
        slide_titles=["A title"],
        text=speech,
        slide_text="",
        cleaned_text="",
        clean_text="",
        slide_description="",
        embed_text=full,
    )


def ids(results):
    return [result.chunk.chunk_id for result in results]


def unrelated_chunks(count):
    # BM25 gives a word that is in half of all chunks (or more) no weight, so tiny test corpora
    # need some chunks that have nothing to do with the question, as a real corpus has.
    return [make_chunk(f"filler{i}", speech=f"filler number{i}", full=f"filler number{i}", start=1000.0 + i) for i in range(count)]


def test_a_text_is_cut_into_lower_case_words_without_the_stop_words():
    assert tokenize("What is the F1-score?") == ["f1", "score"]


def test_the_chunk_with_the_rare_question_word_comes_first():
    chunks = [
        make_chunk("a", full="the model is trained on the data and the data is split"),
        make_chunk("b", full="the bootstrap draws samples with replacement from the data"),
        make_chunk("c", full="the data is cleaned and the data is stored"),
    ]

    results = keyword_search(chunks, "how does the bootstrap work with data", signal="bm25")

    assert ids(results)[0] == "b"
    assert results[0].method == "bm25"


def test_chunks_that_share_no_word_with_the_question_are_left_out():
    chunks = [make_chunk("a", full="cats and dogs"), make_chunk("b", full="bootstrap samples")] + unrelated_chunks(4)

    results = keyword_search(chunks, "bootstrap", signal="bm25")

    assert ids(results) == ["b"]


def test_a_question_made_only_of_stop_words_finds_nothing():
    chunks = [make_chunk("a", full="cats and dogs")]

    assert keyword_search(chunks, "what is the", signal="bm25") == []


def test_no_chunks_give_no_results():
    assert keyword_search([], "bootstrap", signal="bm25") == []


def test_top_k_limits_the_results():
    chunks = [make_chunk(f"c{i}", full=f"bootstrap word{i}") for i in range(5)]

    assert len(keyword_search(chunks, "bootstrap", signal="bm25", top_k=2)) == 2


def test_the_full_signal_reads_the_slide_text_and_the_speech_signal_only_what_was_said():
    # The lecturer's speech was heard as "rock", but the slide says AUROC
    chunks = [make_chunk("a", speech="this is our rock curve", full="AUROC slide this is our rock curve")] + unrelated_chunks(4)

    assert ids(keyword_search(chunks, "what is the AUROC", signal="bm25")) == ["a"]
    assert keyword_search(chunks, "what is the AUROC", signal="bm25_speech") == []


def test_an_unknown_signal_is_refused():
    with pytest.raises(ValueError):
        keyword_search([make_chunk("a", full="x")], "x", signal="nonsense")


def test_keyword_signals_are_told_apart_from_vector_signals():
    assert is_keyword_signal("bm25")
    assert is_keyword_signal("bm25_speech")
    assert not is_keyword_signal("speech")
    assert dense_signals_only(["speech", "full", "bm25"]) == ["speech", "full"]
