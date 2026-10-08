# Tests for the chunk checks in verify.py (check_chunks_data).
#
# Start from a correct tiny lecture (built with build_chunks), prove every check passes,
# then break one thing at a time and prove exactly that check notices.

from src.processing.chunking import build_chunks
from src.schemas.knowledge_object import AlignedSegment, KnowledgeObject
from src.verify import FAIL, PASS, check_chunks_data

MAX_WORDS = 40


def make_slide(start, end, title=""):
    return KnowledgeObject(
        lecture_id="lecture_test",
        start_timestamp=start,
        end_timestamp=end,
        transcript="",
        slide_text="",
        slide_description="",
        image_path=f"frame_{start}.jpg",
        title=title,
    )


def make_segment(start, slide_time):
    words = []
    for i in range(10):
        words.append(f"s{int(start)}_{i}")
    return AlignedSegment(start=start, end=start + 4.0, text=" " + " ".join(words), keyframe_timestamp=slide_time)


def make_good_lecture():
    # Two slides with speech, a blank section title between them, and enough speech for 3 chunks
    slides = [
        make_slide(0.0, 40.0, title="One"),
        make_slide(40.0, 50.0, title="Section"),
        make_slide(50.0, 200.0, title="Two"),
    ]
    segments = [
        make_segment(5.0, 0.0),
        make_segment(15.0, 0.0),
        make_segment(25.0, 0.0),
        make_segment(60.0, 50.0),
        make_segment(70.0, 50.0),
        make_segment(80.0, 50.0),
        make_segment(90.0, 50.0),
        make_segment(100.0, 50.0),
    ]
    chunks = build_chunks(slides, segments, target_words=25, max_words=MAX_WORDS, overlap_segments=1, min_last_words=0)
    return slides, segments, chunks


def run_checks(slides, segments, chunks):
    return check_chunks_data(slides, segments, chunks, MAX_WORDS)


def statuses_by_name(results):
    by_name = {}
    for status, name, detail in results:
        by_name[name] = status
    return by_name


def assert_only_this_check_fails(results, name_fragment):
    failed = []
    for status, name, detail in results:
        if status == FAIL:
            failed.append(name)

    assert len(failed) == 1, f"expected exactly one failure, got {failed}"
    assert name_fragment in failed[0], f"expected {name_fragment!r} to fail, but {failed[0]!r} did"


def test_correct_chunks_pass_every_check():
    slides, segments, chunks = make_good_lecture()

    results = run_checks(slides, segments, chunks)

    failed = []
    for status, name, detail in results:
        if status == FAIL:
            failed.append(name)

    assert failed == []
    assert statuses_by_name(results)["every slide is in some chunk"] == PASS
    assert statuses_by_name(results)["all speech is in some chunk"] == PASS


def test_a_chunk_over_the_maximum_is_caught():
    slides, segments, chunks = make_good_lecture()
    chunks[0].text = chunks[0].text + " extra" * 50

    results = run_checks(slides, segments, chunks)

    assert_only_this_check_fails(results, "over the maximum")


def test_a_slide_that_is_in_no_chunk_is_caught():
    slides, segments, chunks = make_good_lecture()

    # take the blank "Section" slide (starts at 40.0) out of every chunk
    for chunk in chunks:
        if 40.0 in chunk.slide_timestamps:
            chunk.slide_timestamps.remove(40.0)

    results = run_checks(slides, segments, chunks)

    assert_only_this_check_fails(results, "every slide is in some chunk")


def test_lost_speech_is_caught():
    slides, segments, chunks = make_good_lecture()

    # remove one spoken sentence from the text of every chunk that has it
    for chunk in chunks:
        chunk.text = chunk.text.replace("s70_0", "gone")

    results = run_checks(slides, segments, chunks)

    assert_only_this_check_fails(results, "all speech is in some chunk")


def test_a_chunk_reaching_past_its_slides_is_caught():
    slides, segments, chunks = make_good_lecture()
    chunks[-1].end_timestamp = 999.0

    results = run_checks(slides, segments, chunks)

    # the chunk reaches outside its slides, and its end no longer matches the speech it holds
    failed_names = []
    for status, name, detail in results:
        if status == FAIL:
            failed_names.append(name)

    assert "chunk times lie inside their slides" in failed_names


def test_duplicate_ids_are_caught():
    slides, segments, chunks = make_good_lecture()
    chunks[1].chunk_id = chunks[0].chunk_id

    results = run_checks(slides, segments, chunks)

    failed_names = []
    for status, name, detail in results:
        if status == FAIL:
            failed_names.append(name)

    assert "chunk ids are unique" in failed_names


def test_a_broken_neighbour_link_is_caught():
    slides, segments, chunks = make_good_lecture()
    chunks[0].next_chunk_id = "lecture_test_nowhere"

    results = run_checks(slides, segments, chunks)

    assert_only_this_check_fails(results, "neighbours")


def test_no_chunks_at_all_is_caught():
    slides, segments, chunks = make_good_lecture()

    results = run_checks(slides, segments, [])

    assert_only_this_check_fails(results, "chunks exist")
