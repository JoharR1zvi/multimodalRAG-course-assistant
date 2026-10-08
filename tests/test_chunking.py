# Tests for build_chunks(): "cut a lecture into pieces of about the same size, losing nothing".
#
# We use tiny fake lectures: every speech piece has exactly 10 words, and the size limits
# are made small (target 25 words, maximum 40) so that a handful of pieces is enough.

import json

import pytest

from src.processing.chunking import build_chunks, chunk_lecture
from src.schemas.knowledge_object import AlignedSegment, KnowledgeObject


def make_slide(start, end, title="", slide_text="", cleaned_text="", clean_text="", description=""):
    # The smallest valid slide, with the text fields we care about filled in
    return KnowledgeObject(
        lecture_id="lecture_test",
        start_timestamp=start,
        end_timestamp=end,
        transcript="",
        slide_text=slide_text,
        slide_description=description,
        image_path=f"frame_{start}.jpg",
        title=title,
        cleaned_text=cleaned_text,
        clean_text=clean_text,
    )


def make_segment(start, slide_time, word_count=10):
    # A speech piece of exactly word_count words, 4 seconds long.
    # Every word is unique ("s5_0", "s5_1", ...), so we can tell later where it ended up.
    words = []
    for i in range(word_count):
        words.append(f"s{int(start)}_{i}")
    text = " " + " ".join(words)       # Whisper puts a space in front of every piece
    return AlignedSegment(start=start, end=start + 4.0, text=text, keyframe_timestamp=slide_time)


def one_slide_lecture(segment_count):
    # One slide, and segment_count speech pieces of 10 words, starting at 0s, 5s, 10s, ...
    slides = [make_slide(0.0, 1000.0)]
    segments = []
    for i in range(segment_count):
        segments.append(make_segment(i * 5.0, 0.0))
    return slides, segments


# Small limits used by most tests. No merging of a small last chunk unless a test asks for it.
SMALL = dict(target_words=25, max_words=40, overlap_segments=1, min_last_words=0)


def words_of(chunk):
    return chunk.text.split()


def test_chunks_are_sealed_when_they_reach_the_target_and_start_with_the_overlap():
    slides, segments = one_slide_lecture(6)

    chunks = build_chunks(slides, segments, **SMALL)

    # pieces 1-3 (30 words >= 25), then 3-5 (3 is the overlap), then 5-6
    assert len(chunks) == 3
    assert len(words_of(chunks[0])) == 30
    assert len(words_of(chunks[1])) == 30
    assert len(words_of(chunks[2])) == 20

    # the last piece of a chunk is repeated at the start of the next one
    assert words_of(chunks[0])[-10:] == words_of(chunks[1])[:10]
    assert words_of(chunks[1])[-10:] == words_of(chunks[2])[:10]


def test_no_chunk_is_ever_over_the_maximum():
    slides, segments = one_slide_lecture(20)

    # a target nobody can reach: only the maximum limits the size
    chunks = build_chunks(slides, segments, target_words=1000, max_words=25, overlap_segments=1, min_last_words=0)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(words_of(chunk)) <= 25


def test_every_spoken_word_is_in_some_chunk():
    slides, segments = one_slide_lecture(9)

    chunks = build_chunks(slides, segments, **SMALL)

    words_in_lecture = set()
    for segment in segments:
        for word in segment.text.split():
            words_in_lecture.add(word)

    words_in_chunks = set()
    for chunk in chunks:
        for word in words_of(chunk):
            words_in_chunks.add(word)

    assert words_in_chunks == words_in_lecture


def test_the_only_repeated_words_are_the_overlap():
    slides, segments = one_slide_lecture(9)

    chunks = build_chunks(slides, segments, **SMALL)

    total_words_in_chunks = 0
    for chunk in chunks:
        total_words_in_chunks = total_words_in_chunks + len(words_of(chunk))

    # 9 pieces of 10 words, plus one repeated piece (10 words) for every join between chunks
    joins = len(chunks) - 1
    assert total_words_in_chunks == 90 + joins * 10


def test_a_slide_nobody_spoke_during_joins_the_next_chunk_not_the_previous_one():
    slides = [
        make_slide(0.0, 100.0, title="Topic A"),
        make_slide(100.0, 110.0, title="Section title"),     # nobody speaks here
        make_slide(110.0, 200.0, title="Topic B"),
    ]
    segments = [
        make_segment(10.0, 0.0),
        make_segment(20.0, 0.0),
        make_segment(30.0, 0.0),        # 30 words: the first chunk is sealed here
        make_segment(120.0, 110.0),
        make_segment(130.0, 110.0),
    ]

    chunks = build_chunks(slides, segments, **SMALL)

    assert "Section title" not in chunks[0].slide_titles
    assert "Section title" in chunks[1].slide_titles
    assert 100.0 in chunks[1].slide_timestamps


def test_slides_without_speech_at_the_very_end_join_the_last_chunk():
    slides = [
        make_slide(0.0, 100.0, title="Topic A"),
        make_slide(100.0, 110.0, title="Final title"),
    ]
    segments = [make_segment(10.0, 0.0), make_segment(20.0, 0.0)]

    chunks = build_chunks(slides, segments, **SMALL)

    assert "Final title" in chunks[-1].slide_titles


def test_a_tiny_last_chunk_is_merged_into_the_one_before_it():
    slides, segments = one_slide_lecture(6)

    # without merging: 3 chunks, the last one has 10 new words (piece 6)
    unmerged = build_chunks(slides, segments, **SMALL)
    assert len(unmerged) == 3

    # with merging: 10 new words < 50, and 30 + 10 = 40 fits under the maximum of 40
    merged = build_chunks(slides, segments, target_words=25, max_words=40, overlap_segments=1, min_last_words=50)
    assert len(merged) == 2
    assert len(words_of(merged[1])) == 40


def test_a_tiny_last_chunk_stays_separate_when_merging_would_break_the_maximum():
    slides, segments = one_slide_lecture(6)

    chunks = build_chunks(slides, segments, target_words=25, max_words=35, overlap_segments=1, min_last_words=50)

    assert len(chunks) == 3
    for chunk in chunks:
        assert len(words_of(chunk)) <= 35


def test_chunk_times_are_the_exact_times_of_the_speech_inside():
    slides, segments = one_slide_lecture(6)

    chunks = build_chunks(slides, segments, **SMALL)

    # first chunk = pieces at 0s, 5s, 10s; the last one ends 4 seconds after it starts
    assert chunks[0].start_timestamp == 0.0
    assert chunks[0].end_timestamp == 14.0
    # second chunk starts with the overlap piece (the one at 10s) and ends with the piece at 20s
    assert chunks[1].start_timestamp == 10.0
    assert chunks[1].end_timestamp == 24.0


def test_chunk_ids_are_built_from_lecture_and_start_second_and_neighbours_are_linked():
    slides, segments = one_slide_lecture(6)

    chunks = build_chunks(slides, segments, **SMALL)

    assert chunks[0].chunk_id == "lecture_test_0"
    assert chunks[1].chunk_id == "lecture_test_10"
    assert chunks[2].chunk_id == "lecture_test_20"

    assert chunks[0].prev_chunk_id is None
    assert chunks[0].next_chunk_id == "lecture_test_10"
    assert chunks[1].prev_chunk_id == "lecture_test_0"
    assert chunks[1].next_chunk_id == "lecture_test_20"
    assert chunks[2].next_chunk_id is None


def test_building_twice_gives_the_same_ids():
    slides, segments = one_slide_lecture(6)

    first = build_chunks(slides, segments, **SMALL)
    second = build_chunks(slides, segments, **SMALL)

    first_ids = []
    for chunk in first:
        first_ids.append(chunk.chunk_id)
    second_ids = []
    for chunk in second:
        second_ids.append(chunk.chunk_id)

    assert first_ids == second_ids


def test_a_chunk_lists_every_slide_it_covers_and_the_times_lie_inside_them():
    slides = [
        make_slide(0.0, 20.0, title="One", slide_text="raw one"),
        make_slide(20.0, 40.0, title="Two", slide_text="raw two"),
        make_slide(40.0, 100.0, title="Three", slide_text="raw three"),
    ]
    segments = [
        make_segment(5.0, 0.0),
        make_segment(25.0, 20.0),
        make_segment(45.0, 40.0),        # 30 words: sealed here
        make_segment(60.0, 40.0),
    ]

    chunks = build_chunks(slides, segments, **SMALL)

    assert chunks[0].slide_timestamps == [0.0, 20.0, 40.0]
    assert chunks[0].slide_titles == ["One", "Two", "Three"]
    assert chunks[0].image_paths == ["frame_0.0.jpg", "frame_20.0.jpg", "frame_40.0.jpg"]
    assert chunks[0].slide_text == "raw one\nraw two\nraw three"

    # every chunk lies inside the time span of the slides it lists
    slide_ends = {0.0: 20.0, 20.0: 40.0, 40.0: 100.0}
    for chunk in chunks:
        first_slide = chunk.slide_timestamps[0]
        last_slide = chunk.slide_timestamps[-1]
        assert chunk.start_timestamp >= first_slide
        assert chunk.end_timestamp <= slide_ends[last_slide]


def test_the_embedded_text_uses_the_chosen_version_of_the_slide_text():
    slides = [
        make_slide(
            0.0, 100.0,
            slide_text="RAW", cleaned_text="CLEANED", clean_text="VISION",
            description="A DIAGRAM",
        )
    ]
    segments = [make_segment(5.0, 0.0)]

    for source, expected, others in [
        ("slide_text", "RAW", ["CLEANED", "VISION"]),
        ("cleaned_text", "CLEANED", ["RAW", "VISION"]),
        ("clean_text", "VISION", ["RAW", "CLEANED"]),
    ]:
        chunks = build_chunks(slides, segments, slide_text_source=source, **SMALL)
        embed_text = chunks[0].embed_text

        # slide text first, then the description, then the speech
        assert embed_text.startswith(expected)
        assert embed_text.index("A DIAGRAM") > embed_text.index(expected)
        assert embed_text.index("s5_0") > embed_text.index("A DIAGRAM")
        for other in others:
            assert other not in embed_text

    # all three versions stay available on the chunk for experiments
    assert chunks[0].slide_text == "RAW"
    assert chunks[0].cleaned_text == "CLEANED"
    assert chunks[0].clean_text == "VISION"


def test_speech_before_the_first_slide_is_kept_in_the_first_chunk():
    slides = [make_slide(50.0, 100.0)]
    segments = [
        make_segment(1.0, None),         # spoken before any slide
        make_segment(60.0, 50.0),
    ]

    chunks = build_chunks(slides, segments, **SMALL)

    assert len(chunks) == 1
    assert "s1_0" in chunks[0].text
    assert chunks[0].slide_timestamps == [50.0]


def test_speech_that_points_at_a_missing_slide_is_refused_instead_of_lost():
    slides = [make_slide(0.0, 100.0)]
    segments = [make_segment(5.0, 777.0)]

    with pytest.raises(ValueError):
        build_chunks(slides, segments, **SMALL)


def test_an_unknown_slide_text_source_is_refused():
    slides, segments = one_slide_lecture(2)

    with pytest.raises(ValueError):
        build_chunks(slides, segments, slide_text_source="nonsense", **SMALL)


def test_no_speech_means_no_chunks():
    slides = [make_slide(0.0, 100.0)]

    assert build_chunks(slides, [], **SMALL) == []


def test_the_stage_writes_chunks_json_and_skips_when_it_exists_unless_forced(tmp_path):
    slides, segments = one_slide_lecture(6)

    alignment_path = tmp_path / "alignment.json"
    knowledge_path = tmp_path / "knowledge_objects.json"
    output_path = tmp_path / "chunks.json"

    segment_data = []
    for segment in segments:
        segment_data.append(segment.model_dump())

    slide_data = []
    for slide in slides:
        slide_data.append(slide.model_dump())

    with open(alignment_path, "w", encoding="utf-8") as f:
        json.dump(segment_data, f)
    with open(knowledge_path, "w", encoding="utf-8") as f:
        json.dump(slide_data, f)

    chunk_lecture(alignment_path, knowledge_path, output_path)
    assert output_path.exists()

    # Pretend someone edited the file: a second run without force must leave it alone
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[]")
    chunk_lecture(alignment_path, knowledge_path, output_path)
    with open(output_path, encoding="utf-8") as f:
        assert json.load(f) == []

    # With force it is rebuilt
    chunk_lecture(alignment_path, knowledge_path, output_path, force=True)
    with open(output_path, encoding="utf-8") as f:
        assert len(json.load(f)) > 0
