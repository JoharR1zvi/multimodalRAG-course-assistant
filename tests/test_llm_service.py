# Tests for llm_service.py (the prompt, the citation checking, the answer step) and for how
# ask.py shows a source.
#
# Gemini is replaced by a fake function, so no test uses the API or the quota. What matters
# here is what OUR code does with whatever the model says.

import pytest

from src.ask import format_source
from src.generation import llm_service
from src.generation.llm_service import answer_question, build_prompt, extract_citations
from src.schemas.answer import GroundedAnswer
from src.schemas.chunk import Chunk, SearchResult


def make_result(chunk_id, lecture_id="lecture_a", start=0.0, end=100.0, titles=None, text="the speech", clean_text="vision text", cleaned_text="cleaned ocr", description="", score=0.5):
    if titles is None:
        titles = ["A title"]
    chunk = Chunk(
        chunk_id=chunk_id,
        course_id="course_test",
        lecture_id=lecture_id,
        start_timestamp=start,
        end_timestamp=end,
        slide_timestamps=[start],
        image_paths=[f"frame_{start}.jpg"],
        slide_titles=titles,
        text=text,
        slide_text="raw ocr",
        cleaned_text=cleaned_text,
        clean_text=clean_text,
        slide_description=description,
        embed_text="embedded",
    )
    return SearchResult(chunk=chunk, score=score, method="dense")


def three_results():
    return [
        make_result("a_0", lecture_id="lecture_a", start=1074.0, end=1245.0, titles=["First topic"], text="speech one"),
        make_result("b_0", lecture_id="lecture_b", start=65.0, end=300.0, titles=["Second topic"], text="speech two"),
        make_result("a_1", lecture_id="lecture_a", start=5295.0, end=5400.0, titles=["Third topic"], text="speech three"),
    ]


class FakeGemini:
    # Replaces ask_gemini: remembers the prompt it got and answers with a prepared form.
    def __init__(self, answerable=True, answer="An answer [1]."):
        self.answerable = answerable
        self.answer = answer
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return GroundedAnswer(answerable=self.answerable, answer=self.answer)


# ---------- the prompt ----------

def test_the_prompt_has_the_question_and_numbered_excerpts_with_real_lecture_and_time():
    prompt = build_prompt("What is X?", three_results())

    assert "Question: What is X?" in prompt
    assert "[1] lecture_a, 17:54 - 20:45" in prompt
    assert "[2] lecture_b, 1:05 - 5:00" in prompt
    assert "[3] lecture_a, 1:28:15 - 1:30:00" in prompt
    assert "speech one" in prompt
    assert "speech three" in prompt
    # excerpts appear in the order given
    assert prompt.index("[1]") < prompt.index("[2]") < prompt.index("[3]")


def test_the_prompt_uses_the_vision_slide_text_and_the_description():
    results = [make_result("a_0", clean_text="VISION TEXT", cleaned_text="CLEANED OCR", description="A DIAGRAM OF X")]

    prompt = build_prompt("Q", results)

    assert "VISION TEXT" in prompt
    assert "CLEANED OCR" not in prompt
    assert "A DIAGRAM OF X" in prompt
    assert "raw ocr" not in prompt


def test_the_prompt_falls_back_to_the_cleaned_ocr_text_when_there_is_no_vision_text():
    results = [make_result("a_0", clean_text="", cleaned_text="CLEANED OCR")]

    assert "CLEANED OCR" in build_prompt("Q", results)


# ---------- reading the citations ----------

def test_valid_citations_are_kept_and_listed_in_order_of_first_appearance():
    text, cited, invalid = extract_citations("Second idea [3]. First idea [1]. Again [3].", 3)

    assert text == "Second idea [3]. First idea [1]. Again [3]."
    assert cited == [3, 1]
    assert invalid == []


def test_several_numbers_in_one_bracket_are_understood():
    text, cited, invalid = extract_citations("Both agree [1, 2] and also [2,3].", 3)

    assert cited == [1, 2, 3]
    assert invalid == []
    assert text == "Both agree [1, 2] and also [2, 3]."


def test_a_citation_to_an_excerpt_that_does_not_exist_is_removed_and_reported():
    text, cited, invalid = extract_citations("Real claim [1]. Made-up claim [9].", 3)

    assert text == "Real claim [1]. Made-up claim."
    assert cited == [1]
    assert invalid == [9]


def test_in_a_mixed_bracket_only_the_invalid_number_is_removed():
    text, cited, invalid = extract_citations("Claim [2, 7].", 3)

    assert text == "Claim [2]."
    assert cited == [2]
    assert invalid == [7]


def test_zero_is_not_a_valid_excerpt_number():
    text, cited, invalid = extract_citations("Claim [0].", 3)

    assert cited == []
    assert invalid == [0]


def test_text_without_citations_gives_no_citations():
    text, cited, invalid = extract_citations("Just words.", 3)

    assert text == "Just words."
    assert cited == []


# ---------- the whole answer step ----------

def test_the_sources_come_from_the_real_chunks_not_from_the_model(monkeypatch):
    fake = FakeGemini(answer="Claim A [3]. Claim B [1].")
    monkeypatch.setattr(llm_service, "ask_gemini", fake)
    results = three_results()

    final = answer_question("Q?", results)

    assert final.answerable is True
    assert final.source_numbers == [3, 1]
    # the sources are the actual search results with their real lecture and times
    assert final.sources[0].chunk.chunk_id == "a_1"
    assert final.sources[0].chunk.start_timestamp == 5295.0
    assert final.sources[1].chunk.chunk_id == "a_0"
    assert final.warnings == []


def test_the_model_receives_the_numbered_excerpts(monkeypatch):
    fake = FakeGemini()
    monkeypatch.setattr(llm_service, "ask_gemini", fake)

    answer_question("What is X?", three_results())

    assert len(fake.prompts) == 1
    assert "Question: What is X?" in fake.prompts[0]
    assert "[3] lecture_a" in fake.prompts[0]


def test_an_invented_citation_gives_a_warning_and_is_not_a_source(monkeypatch):
    monkeypatch.setattr(llm_service, "ask_gemini", FakeGemini(answer="Claim [1]. Invented [8]."))

    final = answer_question("Q?", three_results())

    assert final.source_numbers == [1]
    assert "[8]" not in final.text
    assert len(final.warnings) == 1
    assert "8" in final.warnings[0]


def test_an_answer_without_any_citation_is_flagged(monkeypatch):
    monkeypatch.setattr(llm_service, "ask_gemini", FakeGemini(answer="A confident answer with no citation."))

    final = answer_question("Q?", three_results())

    assert final.answerable is True
    assert final.sources == []
    assert len(final.warnings) == 1
    assert "no valid citation" in final.warnings[0]


def test_a_question_the_material_does_not_cover_is_reported_as_not_answerable(monkeypatch):
    monkeypatch.setattr(llm_service, "ask_gemini", FakeGemini(answerable=False, answer="The lecture material does not cover this."))

    final = answer_question("Weather in Paris?", three_results())

    assert final.answerable is False
    assert final.sources == []
    assert final.warnings == []
    assert "does not cover" in final.text


def test_no_search_results_means_the_model_is_not_even_asked(monkeypatch):
    def must_not_be_called(prompt):
        raise AssertionError("Gemini was called although nothing was found")

    monkeypatch.setattr(llm_service, "ask_gemini", must_not_be_called)

    final = answer_question("Q?", [])

    assert final.answerable is False
    assert final.sources == []


# ---------- how a source is shown ----------

def test_a_source_line_shows_number_lecture_times_titles_and_image():
    result = make_result("a_0", lecture_id="lecture_a", start=1074.0, end=1245.0, titles=["Same", "Same", "Other"])

    text = format_source(2, result)

    assert text.startswith("[2] lecture_a | 17:54 - 20:45 | Same | Other")
    assert "slide image: frame_1074.0.jpg" in text
