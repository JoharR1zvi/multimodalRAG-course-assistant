# Tests for the evaluation helpers (src/evaluate.py).
#
# Tiny hand-made data only: a few fake chunks, a few fake questions, an in-memory database
# and a fake question embedder, so no model, no real lecture and no Gemini call is needed.

import json
from datetime import datetime

import numpy as np
import pytest

from src.database.vector_store import (
    create_collection_if_missing,
    get_collection_name,
    open_client,
    upsert_chunks,
)
from src.evaluate import (
    chunk_covers_location,
    chunk_is_hit,
    compute_hop_metrics,
    compute_metrics,
    count_hits_at_k,
    count_hops_found,
    find_first_hit_rank,
    find_first_strict_hit_rank,
    get_locations,
    load_eval_set,
    mean_reciprocal_rank,
    mean_top_chunk_seconds,
    overlap_seconds,
    ranges_overlap,
    run_retrieval,
    save_results,
    score_range,
    settings_snapshot,
    split_by_completeness,
    split_by_type,
    summarize_results,
)
from src import evaluate
from src.retrieval import retriever
from src.schemas.answer import FinalAnswer
from src.schemas.chunk import Chunk, SearchResult


def make_chunk(chunk_id, lecture_id="lecture_a", start=0.0, end=100.0):
    return Chunk(
        chunk_id=chunk_id,
        course_id="course_test",
        lecture_id=lecture_id,
        start_timestamp=start,
        end_timestamp=end,
        slide_timestamps=[start],
        image_paths=[f"frame_{start}.jpg"],
        slide_titles=["A title"],
        text="some speech",
        slide_text="",
        cleaned_text="",
        clean_text="",
        slide_description="",
        embed_text="embedded",
    )


def make_result(chunk_id, lecture_id="lecture_a", start=0.0, end=100.0, score=0.5):
    return SearchResult(chunk=make_chunk(chunk_id, lecture_id, start, end), score=score, method="dense")


def make_item(lecture="lecture_a", ranges=None, completeness="full"):
    if ranges is None:
        ranges = [[50, 80]]
    return {
        "id": "q1",
        "question": "a question",
        "answerable": True,
        "lecture": lecture,
        "answer_ranges": ranges,
        "answer_completeness": completeness,
    }


# --- ranges_overlap ---

def test_stretches_that_share_seconds_overlap():
    assert ranges_overlap(0, 100, 50, 80) is True      # one inside the other
    assert ranges_overlap(0, 60, 50, 80) is True       # overlap at the end of the chunk
    assert ranges_overlap(70, 200, 50, 80) is True     # overlap at the start of the chunk
    assert ranges_overlap(0, 1000, 50, 80) is True     # the chunk contains the whole range


def test_stretches_that_only_touch_or_are_apart_do_not_overlap():
    assert ranges_overlap(0, 50, 50, 80) is False      # chunk ends exactly where the range starts
    assert ranges_overlap(80, 200, 50, 80) is False    # chunk starts exactly where the range ends
    assert ranges_overlap(0, 10, 50, 80) is False
    assert ranges_overlap(300, 400, 50, 80) is False


# --- chunk_is_hit ---

def test_a_chunk_is_a_hit_when_lecture_and_time_both_fit():
    chunk = make_chunk("c", lecture_id="lecture_a", start=40, end=90)
    assert chunk_is_hit(chunk, make_item()) is True


def test_the_right_time_in_the_wrong_lecture_is_not_a_hit():
    chunk = make_chunk("c", lecture_id="lecture_b", start=40, end=90)
    assert chunk_is_hit(chunk, make_item(lecture="lecture_a")) is False


def test_the_right_lecture_at_the_wrong_time_is_not_a_hit():
    chunk = make_chunk("c", lecture_id="lecture_a", start=500, end=600)
    assert chunk_is_hit(chunk, make_item()) is False


def test_any_one_of_several_answer_ranges_is_enough():
    item = make_item(ranges=[[50, 80], [1000, 1100]])
    assert chunk_is_hit(make_chunk("c", start=1050, end=1200), item) is True
    assert chunk_is_hit(make_chunk("c", start=500, end=600), item) is False


# --- the stricter hit: the chunk must cover at least half of the answer range ---

def test_the_shared_seconds_of_two_stretches():
    assert overlap_seconds(0, 100, 50, 80) == 30      # one inside the other
    assert overlap_seconds(0, 60, 50, 80) == 10
    assert overlap_seconds(0, 50, 50, 80) == 0        # only touching
    assert overlap_seconds(0, 10, 50, 80) == 0


def test_a_chunk_that_covers_half_of_the_answer_range_is_a_strict_hit():
    location = {"lecture": "lecture_a", "ranges": [[100, 160]]}

    assert chunk_covers_location(make_chunk("c", start=90, end=200), location, 0.5) is True      # all of it
    assert chunk_covers_location(make_chunk("c", start=130, end=300), location, 0.5) is True     # exactly half (30 of 60)
    assert chunk_covers_location(make_chunk("c", start=140, end=300), location, 0.5) is False    # a third (20 of 60)
    assert chunk_covers_location(make_chunk("c", lecture_id="lecture_b", start=90, end=200), location, 0.5) is False


def test_a_big_chunk_that_only_touches_the_answer_is_a_hit_but_not_a_strict_hit():
    item = make_item(ranges=[[100, 160]])
    big_chunk_result = make_result("big", start=150, end=450)       # shares 10 of the 60 seconds

    assert find_first_hit_rank([big_chunk_result], item) == 1
    assert find_first_strict_hit_rank([big_chunk_result], item) is None


def test_the_strict_rank_is_the_position_of_the_first_chunk_that_covers_enough():
    item = make_item(ranges=[[100, 160]])
    results = [
        make_result("touching", start=150, end=450),
        make_result("covering", start=90, end=200),
    ]

    assert find_first_strict_hit_rank(results, item) == 2


def test_the_strict_rank_counts_any_of_the_places_of_a_question():
    item = make_item(lecture="lecture_a", ranges=[[100, 160]])
    item["extra_locations"] = [{"lecture": "lecture_b", "ranges": [[10, 50]]}]

    assert find_first_strict_hit_rank([make_result("c", lecture_id="lecture_b", start=0, end=60)], item) == 1


def test_the_average_length_of_the_best_results():
    rows = [
        {"results": [{"start": 0, "end": 100}]},
        {"results": [{"start": 50, "end": 350}]},
        {"results": []},
    ]

    assert mean_top_chunk_seconds(rows) == 200
    assert mean_top_chunk_seconds([]) == 0.0


# --- other places that also answer a question, and questions that need several places ---

def test_the_places_of_a_question_are_the_main_place_the_extra_places_and_the_hops():
    item = make_item(ranges=[[50, 80]])
    item["extra_locations"] = [{"lecture": "lecture_b", "ranges": [[10, 20]]}]
    item["hops"] = [{"lecture": "lecture_c", "ranges": [[1, 2]]}]

    locations = get_locations(item)

    assert [location["lecture"] for location in locations] == ["lecture_a", "lecture_b", "lecture_c"]


def test_a_question_without_ranges_or_extra_places_has_no_places():
    item = {"id": "u", "answerable": False, "lecture": None, "answer_ranges": []}
    assert get_locations(item) == []
    assert chunk_is_hit(make_chunk("c"), item) is False


def test_a_chunk_at_an_extra_place_is_a_hit_too():
    item = make_item(lecture="lecture_a", ranges=[[50, 80]])
    item["extra_locations"] = [{"lecture": "lecture_b", "ranges": [[300, 400]]}]

    assert chunk_is_hit(make_chunk("c", lecture_id="lecture_b", start=350, end=450), item) is True
    # The extra place counts only in its own lecture
    assert chunk_is_hit(make_chunk("c", lecture_id="lecture_a", start=350, end=450), item) is False


def test_a_chunk_at_any_hop_is_a_hit():
    item = {
        "id": "m", "answerable": True, "lecture": "lecture_a", "answer_ranges": [],
        "hops": [{"lecture": "lecture_a", "ranges": [[0, 50]]}, {"lecture": "lecture_b", "ranges": [[500, 600]]}],
    }
    assert chunk_is_hit(make_chunk("c", lecture_id="lecture_b", start=520, end=620), item) is True
    assert chunk_is_hit(make_chunk("c", lecture_id="lecture_b", start=0, end=100), item) is False


def test_hops_found_counts_the_places_that_have_a_hit_in_the_top_k():
    item = {
        "id": "m", "answerable": True, "lecture": "lecture_a", "answer_ranges": [],
        "hops": [{"lecture": "lecture_a", "ranges": [[0, 50]]}, {"lecture": "lecture_b", "ranges": [[500, 600]]}],
    }
    results = [
        make_result("first", lecture_id="lecture_a", start=10, end=60),     # hits hop 1
        make_result("filler", lecture_id="lecture_a", start=900, end=950),  # hits nothing
        make_result("third", lecture_id="lecture_b", start=520, end=620),   # hits hop 2
    ]

    assert count_hops_found(results, item, 1) == 1
    assert count_hops_found(results, item, 2) == 1
    assert count_hops_found(results, item, 3) == 2


def test_two_chunks_hitting_the_same_hop_count_that_hop_once():
    item = {
        "id": "m", "answerable": True, "lecture": "lecture_a", "answer_ranges": [],
        "hops": [{"lecture": "lecture_a", "ranges": [[0, 50]]}, {"lecture": "lecture_b", "ranges": [[500, 600]]}],
    }
    results = [
        make_result("a", lecture_id="lecture_a", start=0, end=40),
        make_result("b", lecture_id="lecture_a", start=10, end=60),
    ]
    assert count_hops_found(results, item, 2) == 1


def test_hop_metrics_give_the_share_of_complete_questions_and_of_places():
    rows = [
        {"hops_total": 2, "hops_found": {1: 1, 3: 2}},   # complete at k=3
        {"hops_total": 2, "hops_found": {1: 0, 3: 1}},   # half found at k=3
    ]
    metrics = compute_hop_metrics(rows, [1, 3])

    assert metrics["count"] == 2
    assert metrics["all_hops@1"] == pytest.approx(0.0)
    assert metrics["hop_recall@1"] == pytest.approx(0.25)
    assert metrics["all_hops@3"] == pytest.approx(0.5)
    assert metrics["hop_recall@3"] == pytest.approx(0.75)


def test_hop_metrics_of_no_questions_do_not_divide_by_zero():
    metrics = compute_hop_metrics([], [1])
    assert metrics["all_hops@1"] == 0.0
    assert metrics["hop_recall@1"] == 0.0


def test_rows_are_grouped_by_type_in_order_and_untyped_rows_are_left_out():
    rows = [
        {"id": "a", "type": "detail"},
        {"id": "b", "type": "decoy"},
        {"id": "c", "type": "detail"},
        {"id": "d", "type": None},
        {"id": "e"},
    ]
    groups = split_by_type(rows)

    assert list(groups.keys()) == ["detail", "decoy"]
    assert [row["id"] for row in groups["detail"]] == ["a", "c"]


# --- find_first_hit_rank ---

def test_the_rank_is_the_position_of_the_first_hit():
    results = [
        make_result("miss1", start=500, end=600),
        make_result("hit1", start=40, end=90),
        make_result("hit2", start=60, end=70),
    ]
    assert find_first_hit_rank(results, make_item()) == 2


def test_no_hit_gives_none():
    results = [make_result("miss1", start=500, end=600), make_result("miss2", lecture_id="lecture_b", start=40, end=90)]
    assert find_first_hit_rank(results, make_item()) is None


def test_no_results_gives_none():
    assert find_first_hit_rank([], make_item()) is None


# --- metrics ---

def test_hits_at_k_counts_ranks_up_to_k_and_ignores_misses():
    ranks = [1, 3, None, 5, 2]
    assert count_hits_at_k(ranks, 1) == 1
    assert count_hits_at_k(ranks, 3) == 3
    assert count_hits_at_k(ranks, 5) == 4
    assert count_hits_at_k(ranks, 10) == 4


def test_mrr_is_the_average_of_one_over_rank_and_a_miss_counts_zero():
    # (1/1 + 1/2 + 0) / 3 = 0.5
    assert mean_reciprocal_rank([1, 2, None]) == pytest.approx(0.5)
    assert mean_reciprocal_rank([None, None]) == 0.0


def test_metrics_of_no_questions_do_not_divide_by_zero():
    metrics = compute_metrics([], [1, 3])
    assert metrics["count"] == 0
    assert metrics["hit@1"] == 0.0
    assert metrics["mrr"] == 0.0


def test_compute_metrics_gives_shares_counts_and_mrr():
    metrics = compute_metrics([1, 2, None, 4], [1, 3, 5])

    assert metrics["count"] == 4
    assert metrics["hits@1"] == 1
    assert metrics["hit@1"] == pytest.approx(0.25)
    assert metrics["hits@3"] == 2
    assert metrics["hit@3"] == pytest.approx(0.5)
    assert metrics["hits@5"] == 3
    assert metrics["hit@5"] == pytest.approx(0.75)
    # (1 + 1/2 + 0 + 1/4) / 4
    assert metrics["mrr"] == pytest.approx(0.4375)


def test_questions_are_split_into_full_and_partial():
    rows = [
        {"id": "a", "completeness": "full"},
        {"id": "b", "completeness": "partial"},
        {"id": "c", "completeness": "full"},
    ]
    full_rows, partial_rows = split_by_completeness(rows)

    assert [row["id"] for row in full_rows] == ["a", "c"]
    assert [row["id"] for row in partial_rows] == ["b"]


def test_score_range_gives_lowest_and_highest_best_score():
    rows = [{"top_score": 0.6}, {"top_score": 0.4}, {"top_score": None}, {"top_score": 0.8}]
    assert score_range(rows) == (0.4, 0.8)
    assert score_range([{"top_score": None}]) is None
    assert score_range([]) is None


def test_the_summary_marks_which_results_are_hits():
    results = [make_result("miss", start=500, end=600, score=0.7), make_result("hit", start=40, end=90, score=0.6)]
    summary = summarize_results(results, make_item())

    assert [entry["rank"] for entry in summary] == [1, 2]
    assert [entry["hit"] for entry in summary] == [False, True]
    assert summary[1]["chunk_id"] == "hit"


def test_the_summary_without_an_item_has_no_hit_flag():
    summary = summarize_results([make_result("c")], None)
    assert "hit" not in summary[0]


# --- answers: a failing Gemini call must not end the whole run ---

def make_final_answer(question, answerable=True, sources=None, coverage=None):
    if sources is None:
        sources = []
    if coverage is None:
        if answerable:
            coverage = "full"
        else:
            coverage = "none"
    return FinalAnswer(
        question=question,
        answerable=answerable,
        coverage=coverage,
        text="an answer",
        sources=sources,
        source_numbers=list(range(1, len(sources) + 1)),
        warnings=[],
    )


@pytest.fixture
def no_waiting(monkeypatch):
    # The tests must not really wait between tries
    waits = []
    monkeypatch.setattr(evaluate.time, "sleep", lambda seconds: waits.append(seconds))
    return waits


def test_a_failing_answer_call_is_tried_again_after_waiting(monkeypatch, no_waiting):
    calls = []

    def flaky_answer(question, results):
        calls.append(question)
        if len(calls) < 3:
            raise RuntimeError("high demand")
        return make_final_answer(question)

    monkeypatch.setattr(evaluate, "answer_question", flaky_answer)

    final = evaluate.answer_with_patience("q", [])

    assert final is not None
    assert len(calls) == 3
    assert no_waiting == evaluate.ANSWER_RETRY_WAITS[:2]


def test_the_answer_call_gives_up_after_the_last_wait(monkeypatch, no_waiting):
    calls = []

    def always_failing(question, results):
        calls.append(question)
        raise RuntimeError("high demand")

    monkeypatch.setattr(evaluate, "answer_question", always_failing)

    assert evaluate.answer_with_patience("q", []) is None
    # the first try plus one try after each wait
    assert len(calls) == len(evaluate.ANSWER_RETRY_WAITS) + 1
    assert no_waiting == evaluate.ANSWER_RETRY_WAITS


def test_a_question_whose_answer_fails_is_recorded_and_the_run_goes_on(monkeypatch, no_waiting):
    items = [
        {"id": "a1", "question": "first", "answerable": True, "lecture": "lecture_a", "answer_ranges": [[50, 80]]},
        {"id": "a2", "question": "second", "answerable": True, "lecture": "lecture_a", "answer_ranges": [[50, 80]]},
        {"id": "u1", "question": "third", "answerable": False, "lecture": None, "answer_ranges": []},
    ]
    hit_result = make_result("hit", start=40, end=90)
    all_results = {"a1": [hit_result], "a2": [hit_result], "u1": [hit_result]}
    answerable_rows = [{"id": "a1", "completeness": "full"}, {"id": "a2", "completeness": "full"}]
    unanswerable_rows = [{"id": "u1"}]

    def answer(question, results):
        if question == "first":
            raise RuntimeError("high demand")
        if question == "second":
            return make_final_answer(question, answerable=True, sources=[hit_result])
        return make_final_answer(question, answerable=False)

    monkeypatch.setattr(evaluate, "answer_question", answer)

    totals = evaluate.run_answers(items, all_results, answerable_rows, unanswerable_rows)

    assert answerable_rows[0]["generated_answerable"] is None
    assert answerable_rows[0]["cites_right_place"] is False
    assert answerable_rows[1]["generated_answerable"] is True
    assert answerable_rows[1]["cites_right_place"] is True
    assert unanswerable_rows[0]["generated_answerable"] is False
    assert totals["answer_errors"] == 1
    assert totals["answerable_citing_right_place"] == 1
    assert totals["unanswerable_refused"] == 1
    assert totals["answerable_wrongly_refused"] == 0


def test_the_totals_say_whether_partly_answered_questions_get_partial_and_full_ones_do_not(monkeypatch, no_waiting):
    items = [
        {"id": "p1", "question": "part one", "answerable": True, "lecture": "lecture_a", "answer_ranges": [[50, 80]]},
        {"id": "p2", "question": "part two", "answerable": True, "lecture": "lecture_a", "answer_ranges": [[50, 80]]},
        {"id": "f1", "question": "full one", "answerable": True, "lecture": "lecture_a", "answer_ranges": [[50, 80]]},
        {"id": "f2", "question": "full two", "answerable": True, "lecture": "lecture_a", "answer_ranges": [[50, 80]]},
    ]
    result = make_result("c", start=40, end=90)
    all_results = {"p1": [result], "p2": [result], "f1": [result], "f2": [result]}
    answerable_rows = [
        {"id": "p1", "completeness": "partial"},
        {"id": "p2", "completeness": "partial"},
        {"id": "f1", "completeness": "full"},
        {"id": "f2", "completeness": "full"},
    ]

    # p1 is marked partial (right), p2 is marked full (a miss), f1 is marked partial (hedging), f2 full (right)
    coverage_by_question = {"part one": "partial", "part two": "full", "full one": "partial", "full two": "full"}

    def answer(question, results):
        return make_final_answer(question, answerable=True, sources=[results[0]], coverage=coverage_by_question[question])

    monkeypatch.setattr(evaluate, "answer_question", answer)

    totals = evaluate.run_answers(items, all_results, answerable_rows, [])

    assert totals["partial_questions_total"] == 2
    assert totals["partial_questions_marked_partial"] == 1
    assert totals["full_questions_total"] == 2
    assert totals["full_questions_marked_partial"] == 1
    assert answerable_rows[0]["generated_coverage"] == "partial"


def test_a_failed_answer_is_left_out_of_the_partial_counts(monkeypatch, no_waiting):
    items = [{"id": "p1", "question": "part one", "answerable": True, "lecture": "lecture_a", "answer_ranges": [[50, 80]]}]
    all_results = {"p1": [make_result("c", start=40, end=90)]}
    answerable_rows = [{"id": "p1", "completeness": "partial"}]

    def failing(question, results):
        raise RuntimeError("high demand")

    monkeypatch.setattr(evaluate, "answer_question", failing)

    totals = evaluate.run_answers(items, all_results, answerable_rows, [])

    assert answerable_rows[0]["generated_coverage"] is None
    assert totals["partial_questions_total"] == 0
    assert totals["answer_errors"] == 1


# --- files ---

def test_the_eval_set_is_read_from_a_file(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps([make_item()]), encoding="utf-8")

    items = load_eval_set(str(path))

    assert len(items) == 1
    assert items[0]["lecture"] == "lecture_a"


def test_results_are_saved_next_to_the_eval_file(tmp_path):
    path = save_results(str(tmp_path), {"hello": "world"})

    assert path.parent == tmp_path
    assert path.name.startswith("results_")
    assert json.loads(path.read_text(encoding="utf-8")) == {"hello": "world"}


def test_two_results_saved_in_the_same_second_do_not_overwrite_each_other(tmp_path, monkeypatch):
    # Freeze the clock so both saves get the same second
    class FrozenClock:
        @staticmethod
        def now():
            return datetime(2026, 10, 8, 12, 0, 0)

    monkeypatch.setattr("src.evaluate.datetime", FrozenClock)

    first = save_results(str(tmp_path), {"run": "first"})
    second = save_results(str(tmp_path), {"run": "second"})

    assert first != second
    assert json.loads(first.read_text(encoding="utf-8")) == {"run": "first"}
    assert json.loads(second.read_text(encoding="utf-8")) == {"run": "second"}


def test_the_settings_snapshot_names_the_settings_that_matter():
    snapshot = settings_snapshot(7)

    assert snapshot["top_k"] == 7
    assert "chunk_target_words" in snapshot
    assert "chunk_slide_text_source" in snapshot
    assert "chunk_include_description" in snapshot
    assert "qdrant_path" in snapshot
    assert "embedding_model" in snapshot
    assert "embedding_max_tokens" in snapshot


# --- the whole search part, on a tiny in-memory database ---

@pytest.fixture
def database(monkeypatch):
    # Every question is embedded as "points at the second axis". The chunk of lecture_a that
    # covers seconds 100-200 points exactly there, the others point elsewhere.
    monkeypatch.setattr(retriever, "get_model_slug", lambda: "fake-model")
    monkeypatch.setattr(retriever, "embed_query", lambda question: np.array([0.0, 1.0, 0.0]))

    client = open_client(":memory:")
    collection = get_collection_name("fake-model")
    create_collection_if_missing(client, collection, 3)

    chunks = [
        make_chunk("lecture_a_0", lecture_id="lecture_a", start=0, end=100),
        make_chunk("lecture_a_100", lecture_id="lecture_a", start=100, end=200),
        make_chunk("lecture_b_0", lecture_id="lecture_b", start=0, end=100),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.1, 0.9, 0.0]]
    upsert_chunks(client, collection, chunks, vectors)

    yield client
    client.close()


def test_run_retrieval_gives_ranks_for_answerable_and_scores_for_unanswerable(database):
    items = [
        # The best result (lecture_a_100) covers 100-200, so this question is a hit at rank 1
        {"id": "hit1", "question": "q", "answerable": True, "lecture": "lecture_a",
         "answer_ranges": [[120, 150]], "answer_completeness": "full"},
        # The answer is at 0-50 of lecture_a; that chunk is the weakest match, so rank 3
        {"id": "hit3", "question": "q", "answerable": True, "lecture": "lecture_a",
         "answer_ranges": [[10, 50]], "answer_completeness": "partial"},
        # No chunk covers seconds 5000-5100
        {"id": "miss", "question": "q", "answerable": True, "lecture": "lecture_a",
         "answer_ranges": [[5000, 5100]], "answer_completeness": "full"},
        {"id": "u1", "question": "q", "answerable": False, "lecture": None,
         "answer_ranges": [], "answer_completeness": None},
    ]

    answerable_rows, unanswerable_rows, all_results = run_retrieval(items, 3, database)

    assert [row["rank"] for row in answerable_rows] == [1, 3, None]
    # these chunks are 100 s long and the answers are well inside them, so strict = plain here
    assert [row["strict_rank"] for row in answerable_rows] == [1, 3, None]
    assert [row["completeness"] for row in answerable_rows] == ["full", "partial", "full"]
    assert len(unanswerable_rows) == 1
    assert unanswerable_rows[0]["id"] == "u1"
    assert unanswerable_rows[0]["top_score"] == pytest.approx(1.0)
    assert len(all_results["hit1"]) == 3


def test_run_retrieval_counts_the_hops_of_a_multi_hop_question_and_keeps_the_type(database):
    item = {
        "id": "m1", "type": "multi_hop", "question": "q", "answerable": True, "lecture": "lecture_a",
        "answer_ranges": [],
        # lecture_a_100 is the best match (rank 1), lecture_b_0 the second (rank 2)
        "hops": [{"lecture": "lecture_a", "ranges": [[120, 150]]}, {"lecture": "lecture_b", "ranges": [[10, 20]]}],
    }

    answerable_rows, _, _ = run_retrieval([item], 3, database)
    row = answerable_rows[0]

    assert row["type"] == "multi_hop"
    assert row["rank"] == 1
    assert row["hops_total"] == 2
    assert row["hops_found"] == {1: 1, 3: 2}


def test_a_smaller_top_k_turns_a_late_hit_into_a_miss(database):
    item = {"id": "hit3", "question": "q", "answerable": True, "lecture": "lecture_a",
            "answer_ranges": [[10, 50]], "answer_completeness": "full"}

    answerable_rows, _, _ = run_retrieval([item], 2, database)

    # The hit is at rank 3, outside the top 2
    assert answerable_rows[0]["rank"] is None


def test_the_kind_of_search_is_named_from_the_signals():
    assert evaluate.describe_retrieval(["full"]) == "dense"
    assert evaluate.describe_retrieval(["speech", "full"]) == "dense fusion"
    assert evaluate.describe_retrieval(["bm25"]) == "keyword"
    assert evaluate.describe_retrieval(["speech", "full", "bm25"]) == "dense + keyword fusion"
