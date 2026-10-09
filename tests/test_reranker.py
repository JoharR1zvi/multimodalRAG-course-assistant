# Tests for the reranker (src/retrieval/reranker.py) and its place in retrieve().
# A fake cross-encoder replaces the real model: it scores a chunk by hand-made rules, so the real
# model (2 GB) is never loaded.

import numpy as np

from src.database.vector_store import create_collection_if_missing, get_collection_name, open_client, upsert_chunks
from src.retrieval import reranker, retriever
from src.retrieval.reranker import rerank_results
from src.retrieval.retriever import retrieve
from src.schemas.chunk import Chunk, SearchResult


def make_chunk(chunk_id, embed_text="plain", start=0.0):
    return Chunk(
        chunk_id=chunk_id,
        course_id="course_test",
        lecture_id="lecture_a",
        start_timestamp=start,
        end_timestamp=start + 100.0,
        slide_timestamps=[start],
        image_paths=[f"frame_{start}.jpg"],
        slide_titles=["A title"],
        text="speech",
        slide_text="",
        cleaned_text="",
        clean_text="",
        slide_description="",
        embed_text=embed_text,
    )


def make_result(chunk_id, embed_text="plain", score=0.5, start=0.0):
    return SearchResult(chunk=make_chunk(chunk_id, embed_text, start), score=score, method="fusion")


class FakeCrossEncoder:
    # Gives 0.9 to a chunk whose text has the word "answer", 0.1 to the others. Remembers what it was given.
    def __init__(self):
        self.pairs_seen = []

    def predict(self, pairs, batch_size, show_progress_bar):
        self.pairs_seen.append(list(pairs))
        scores = []
        for question, text in pairs:
            if "answer" in text:
                scores.append(0.9)
            else:
                scores.append(0.1)
        return np.array(scores)


def ids(results):
    return [result.chunk.chunk_id for result in results]


def test_the_chunk_the_cross_encoder_likes_best_moves_to_the_front(monkeypatch):
    fake = FakeCrossEncoder()
    monkeypatch.setattr(reranker, "load_reranker", lambda: fake)
    candidates = [make_result("a"), make_result("b"), make_result("c", embed_text="has the answer")]

    reranked = rerank_results("a question", candidates, top_k=3)

    assert ids(reranked) == ["c", "a", "b"]
    assert reranked[0].method == "rerank"
    assert reranked[0].score == 0.9


def test_the_model_reads_the_question_with_the_full_text_of_every_candidate(monkeypatch):
    fake = FakeCrossEncoder()
    monkeypatch.setattr(reranker, "load_reranker", lambda: fake)

    rerank_results("the question", [make_result("a", embed_text="text of a"), make_result("b", embed_text="text of b")], top_k=2)

    assert fake.pairs_seen == [[("the question", "text of a"), ("the question", "text of b")]]


class ScoresByText:
    # A fake cross-encoder with a score written down for each text
    def __init__(self, scores):
        self.scores = scores

    def predict(self, pairs, batch_size, show_progress_bar):
        return np.array([self.scores[text] for question, text in pairs])


def test_blend_mode_merges_the_first_search_order_with_the_reranker_order(monkeypatch):
    # First search order: a, mid, b. Reranker order: b, a, mid.
    # replace -> b, a, mid.   blend -> a (1st and 2nd) edges out b (3rd and 1st): a, b, mid.
    monkeypatch.setattr(reranker, "load_reranker", lambda: ScoresByText({"a": 0.5, "mid": 0.1, "b": 0.9}))
    candidates = [make_result("a", embed_text="a"), make_result("mid", embed_text="mid"), make_result("b", embed_text="b")]

    assert ids(rerank_results("q", candidates, top_k=3, mode="replace")) == ["b", "a", "mid"]
    assert ids(rerank_results("q", candidates, top_k=3, mode="blend")) == ["a", "b", "mid"]


def test_blend_mode_still_shows_the_reranker_score(monkeypatch):
    monkeypatch.setattr(reranker, "load_reranker", lambda: ScoresByText({"a": 0.5, "mid": 0.1, "b": 0.9}))
    candidates = [make_result("a", embed_text="a"), make_result("mid", embed_text="mid"), make_result("b", embed_text="b")]

    blended = rerank_results("q", candidates, top_k=3, mode="blend")

    assert blended[0].score == 0.5
    assert blended[0].method == "rerank"


def test_only_the_best_top_k_are_returned(monkeypatch):
    monkeypatch.setattr(reranker, "load_reranker", lambda: FakeCrossEncoder())
    candidates = [make_result("a"), make_result("b", embed_text="answer"), make_result("c", embed_text="answer")]

    assert ids(rerank_results("q", candidates, top_k=2)) == ["b", "c"]


def test_equal_scores_keep_the_order_of_the_first_search(monkeypatch):
    monkeypatch.setattr(reranker, "load_reranker", lambda: FakeCrossEncoder())
    candidates = [make_result("first"), make_result("second"), make_result("third")]

    assert ids(rerank_results("q", candidates, top_k=3)) == ["first", "second", "third"]


def test_no_candidates_give_no_results_and_no_model_is_loaded(monkeypatch):
    def must_not_load():
        raise AssertionError("the model must not be loaded when there is nothing to rerank")

    monkeypatch.setattr(reranker, "load_reranker", must_not_load)

    assert rerank_results("q", [], top_k=5) == []


def test_retrieve_hands_more_candidates_to_the_reranker_than_it_returns(monkeypatch):
    # Five chunks. By vector the "answer" chunk is LAST (fifth), so a plain top 2 misses it.
    # With reranking the first search fetches all of them and the reranker brings it to the front.
    monkeypatch.setattr(retriever, "get_model_slug", lambda: "fake-model")
    monkeypatch.setattr(retriever, "embed_query", lambda question: np.array([1.0, 0.0, 0.0]))
    monkeypatch.setattr(reranker, "load_reranker", lambda: FakeCrossEncoder())

    client = open_client(":memory:")
    collection = get_collection_name("fake-model")
    create_collection_if_missing(client, collection, 3)
    chunks = [make_chunk(f"c{i}", start=float(i)) for i in range(4)] + [make_chunk("target", embed_text="has the answer", start=10.0)]
    vectors = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0], [0.1, 0.9, 0.0]]
    upsert_chunks(client, collection, chunks, vectors)

    plain = retrieve("q", top_k=2, client=client, rerank=False)
    reranked = retrieve("q", top_k=2, client=client, rerank=True)

    assert "target" not in ids(plain)
    assert ids(reranked)[0] == "target"
    assert len(reranked) == 2
    client.close()
