# Tests for the web page's backend (src/api.py). The search and the answer step are replaced by
# fakes, so no model, no database and no Gemini call is used.

import json

import pytest
from fastapi.testclient import TestClient

from src import api
from src.schemas.answer import FinalAnswer
from src.schemas.chunk import Chunk, SearchResult


def make_chunk(chunk_id="lecture_a_0", lecture_id="lecture_a", start=75.0, end=200.0, image="data\\processed\\lecture_a\\keyframes\\frame_75.00.jpg"):
    return Chunk(
        chunk_id=chunk_id,
        course_id="course_test",
        lecture_id=lecture_id,
        start_timestamp=start,
        end_timestamp=end,
        slide_timestamps=[start],
        image_paths=[image] if image else [],
        slide_titles=["Title one", "Title one", "Title two"],
        text="The lecturer says something about the topic.",
        slide_text="",
        cleaned_text="",
        clean_text="",
        slide_description="",
        embed_text="embedded",
    )


def make_result(**kwargs):
    return SearchResult(chunk=make_chunk(**kwargs), score=0.6234, method="fusion")


def make_final(coverage="full", text="An answer [1].", sources=None, warnings=None):
    if sources is None:
        sources = [make_result()]
    return FinalAnswer(
        question="q",
        answerable=coverage != "none",
        coverage=coverage,
        text=text,
        sources=sources,
        source_numbers=list(range(1, len(sources) + 1)),
        warnings=warnings or [],
    )


@pytest.fixture
def processed(tmp_path, monkeypatch):
    # A data/processed folder with one searchable lecture (with a slide picture) and one without chunks
    lecture = tmp_path / "lecture_a"
    (lecture / "keyframes").mkdir(parents=True)
    (lecture / "chunks.json").write_text(json.dumps([]), encoding="utf-8")
    (lecture / "keyframes" / "frame_75.00.jpg").write_bytes(b"not really a picture")
    (tmp_path / "lecture_b").mkdir()
    monkeypatch.setattr(api, "PROCESSED_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def client(processed):
    return TestClient(api.app)


def fake_steps(monkeypatch, final=None, results=None, calls=None):
    # Replaces the search and the answer step; `calls` collects what the search was asked
    if results is None:
        results = [make_result()]
    if final is None:
        final = make_final()
    if calls is None:
        calls = []

    def fake_retrieve(question, top_k=5, lecture_id=None, **kwargs):
        calls.append({"question": question, "top_k": top_k, "lecture_id": lecture_id})
        return results

    monkeypatch.setattr(api, "retrieve", fake_retrieve)
    monkeypatch.setattr(api, "answer_question", lambda question, found: final)
    return calls


def test_the_page_is_served(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "<title>Course assistant</title>" in response.text


def test_only_lectures_with_chunks_are_listed(client):
    assert client.get("/api/lectures").json() == {"lectures": ["lecture_a"]}


def test_an_answer_comes_back_with_its_sources_as_the_page_needs_them(client, monkeypatch):
    calls = fake_steps(monkeypatch)

    response = client.post("/api/ask", json={"question": "  What is it?  ", "lecture": "lecture_a", "top": 4})
    body = response.json()

    assert response.status_code == 200
    assert calls == [{"question": "What is it?", "top_k": 4, "lecture_id": "lecture_a"}]
    assert body["mode"] == "answer"
    assert body["text"] == "An answer [1]."
    assert body["coverage"] == "full"
    assert body["banner"] == ""

    source = body["sources"][0]
    assert source["number"] == 1
    assert source["lecture_id"] == "lecture_a"
    assert (source["start_label"], source["end_label"]) == ("1:15", "3:20")
    assert source["titles"] == ["Title one", "Title two"]
    assert source["score"] == 0.623
    assert source["excerpt"].startswith("The lecturer says")
    # a path written with backslashes on Windows becomes a page address with only the file name
    assert source["image_url"] == "/slides/lecture_a/frame_75.00.jpg"


def test_a_question_the_lectures_do_not_cover_shows_the_banner_and_the_closest_passages(client, monkeypatch):
    fake_steps(monkeypatch, final=make_final(coverage="none", text="Not covered.", sources=[]), results=[make_result(), make_result(chunk_id="other", start=500.0)])

    body = client.post("/api/ask", json={"question": "Something else?"}).json()

    assert body["answerable"] is False
    assert body["banner"] == "NOT FOUND IN THE COURSE MATERIAL"
    assert body["sources"] == []
    assert len(body["closest"]) == 2


def test_a_partly_covered_answer_has_its_own_banner(client, monkeypatch):
    fake_steps(monkeypatch, final=make_final(coverage="partial"))

    body = client.post("/api/ask", json={"question": "Half?"}).json()

    assert body["banner"].startswith("PARTLY COVERED")
    assert body["closest"] == []


def test_warnings_of_the_answer_step_are_passed_on(client, monkeypatch):
    fake_steps(monkeypatch, final=make_final(warnings=["1 sentence(s) in the answer have no citation"]))

    body = client.post("/api/ask", json={"question": "Q?"}).json()

    assert body["warnings"] == ["1 sentence(s) in the answer have no citation"]


def test_search_only_mode_writes_no_answer(client, monkeypatch):
    fake_steps(monkeypatch)

    def must_not_be_called(question, found):
        raise AssertionError("no answer should be written in search-only mode")

    monkeypatch.setattr(api, "answer_question", must_not_be_called)

    body = client.post("/api/ask", json={"question": "Q?", "answer": False}).json()

    assert body["mode"] == "search"
    assert len(body["closest"]) == 1


def test_an_empty_question_is_refused(client, monkeypatch):
    fake_steps(monkeypatch)

    assert client.post("/api/ask", json={"question": "   "}).status_code == 400
    assert client.post("/api/ask", json={"question": "x" * 1001}).status_code == 422


def test_an_unknown_lecture_is_refused(client, monkeypatch):
    fake_steps(monkeypatch)

    response = client.post("/api/ask", json={"question": "Q?", "lecture": "../secrets"})

    assert response.status_code == 400


def test_a_failing_search_is_reported_as_unavailable(client, monkeypatch):
    def broken_search(question, top_k=5, lecture_id=None, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(api, "retrieve", broken_search)

    response = client.post("/api/ask", json={"question": "Q?"})

    assert response.status_code == 503
    assert "database" in response.json()["detail"]


def test_a_failing_answer_service_is_reported_as_a_bad_gateway(client, monkeypatch):
    fake_steps(monkeypatch)

    def broken_answer(question, found):
        raise RuntimeError("high demand")

    monkeypatch.setattr(api, "answer_question", broken_answer)

    response = client.post("/api/ask", json={"question": "Q?"})

    assert response.status_code == 502
    assert "Gemini" in response.json()["detail"]


def test_a_slide_picture_is_served(client):
    response = client.get("/slides/lecture_a/frame_75.00.jpg")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_only_slide_pictures_can_be_fetched(client, processed):
    # chunks.json exists in the lecture folder, but the route only serves .jpg files from keyframes/
    assert client.get("/slides/lecture_a/chunks.json").status_code == 404
    assert client.get("/slides/lecture_a/frame_999.00.jpg").status_code == 404
    assert client.get("/slides/lecture_x/frame_75.00.jpg").status_code == 404
    assert client.get("/slides/lecture_a/..%2Fchunks.json").status_code == 404


def test_the_page_never_puts_the_answer_in_as_html():
    # The answer text comes from a model and the lectures, so it must only be added as text.
    page = (api.WEB_DIR / "index.html").read_text(encoding="utf-8")

    assert "innerHTML" not in page
    assert "document.write" not in page
