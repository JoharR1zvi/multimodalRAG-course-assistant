# Tests for align(): "for each piece of speech, which slide was on screen?"
#
# align() reads two files and writes one. So each test writes tiny fake input files into a
# temporary folder (pytest hands us one, called tmp_path, and removes it afterwards),
# runs align(), and reads the output file back.

import json

from src.processing.alignment import align


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def make_keyframe(timestamp):
    # The smallest valid slide record
    return {"timestamp": timestamp, "image_path": f"frame_{timestamp}.jpg", "text": ""}


def make_segment(start):
    # A piece of speech that starts at `start` and lasts 2 seconds
    return {"start": start, "end": start + 2.0, "text": f"speech at {start}"}


def run_align(tmp_path, keyframe_times, segment_starts):
    # Writes the fake input files, runs align(), and returns the list of results
    transcript_path = tmp_path / "transcript.json"
    visual_path = tmp_path / "visual_metadata.json"
    output_path = tmp_path / "alignment.json"

    keyframes = []
    for timestamp in keyframe_times:
        keyframes.append(make_keyframe(timestamp))

    segments = []
    for start in segment_starts:
        segments.append(make_segment(start))

    write_json(transcript_path, segments)
    write_json(visual_path, keyframes)

    align(transcript_path, visual_path, output_path)

    with open(output_path, encoding="utf-8") as f:
        return json.load(f)


def slide_of_each_segment(results):
    # Just the "which slide" answer for every piece of speech
    slides = []
    for result in results:
        slides.append(result["keyframe_timestamp"])
    return slides


def test_speech_gets_the_slide_that_was_showing(tmp_path):
    # Slides appear at 0 s and 100 s
    results = run_align(tmp_path, [0.0, 100.0], [5.0, 99.0, 100.0, 150.0])

    # 99 s is still the first slide. At exactly 100 s the second slide has just appeared.
    assert slide_of_each_segment(results) == [0.0, 0.0, 100.0, 100.0]


def test_speech_before_the_first_slide_has_no_slide(tmp_path):
    results = run_align(tmp_path, [10.0, 50.0], [5.0, 10.0, 20.0])

    assert slide_of_each_segment(results) == [None, 10.0, 10.0]


def test_slides_listed_out_of_order_are_sorted_first(tmp_path):
    results = run_align(tmp_path, [100.0, 0.0], [5.0, 150.0])

    assert slide_of_each_segment(results) == [0.0, 100.0]


def test_the_speech_text_and_times_are_kept(tmp_path):
    results = run_align(tmp_path, [0.0], [5.0])

    assert results[0]["start"] == 5.0
    assert results[0]["end"] == 7.0
    assert results[0]["text"] == "speech at 5.0"


def test_an_existing_output_is_not_redone_unless_forced(tmp_path):
    transcript_path = tmp_path / "transcript.json"
    visual_path = tmp_path / "visual_metadata.json"
    output_path = tmp_path / "alignment.json"

    write_json(transcript_path, [make_segment(5.0)])
    write_json(visual_path, [make_keyframe(0.0)])

    # Pretend an earlier run already made an output file
    write_json(output_path, ["old result"])

    # Without force, align() must leave it alone
    align(transcript_path, visual_path, output_path)

    with open(output_path, encoding="utf-8") as f:
        assert json.load(f) == ["old result"]

    # With force, it must be replaced by a real result
    align(transcript_path, visual_path, output_path, force=True)

    with open(output_path, encoding="utf-8") as f:
        new_result = json.load(f)

    assert len(new_result) == 1
    assert new_result[0]["keyframe_timestamp"] == 0.0
