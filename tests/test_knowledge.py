# Tests for build_knowledge_objects(): "one record per slide, with everything said during it".
#
# Like the alignment tests: write tiny fake input files into a temporary folder,
# run the function, read the result back.

import json

from src.processing.knowledge import build_knowledge_objects


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def make_slide(timestamp, **extra_fields):
    # The smallest valid slide record. Extra fields (description, title, ...) can be added.
    slide = {"timestamp": timestamp, "image_path": f"frame_{timestamp}.jpg", "text": f"ocr {timestamp}"}
    slide.update(extra_fields)
    return slide


def make_speech(start, end, text, slide_timestamp):
    # A piece of speech that alignment.py has already matched to a slide.
    # slide_timestamp is None for speech that came before the first slide.
    return {"start": start, "end": end, "text": text, "keyframe_timestamp": slide_timestamp}


def run_knowledge(tmp_path, slides, speech_pieces):
    alignment_path = tmp_path / "alignment.json"
    visual_path = tmp_path / "visual_metadata.json"

    # The lecture id is taken from the NAME OF THE FOLDER the output goes into
    output_path = tmp_path / "lecture_test" / "knowledge_objects.json"

    write_json(alignment_path, speech_pieces)
    write_json(visual_path, slides)

    build_knowledge_objects(alignment_path, visual_path, output_path)

    with open(output_path, encoding="utf-8") as f:
        return json.load(f)


def test_speech_is_grouped_under_its_slide_and_leading_spaces_are_removed(tmp_path):
    slides = [make_slide(0.0), make_slide(100.0)]
    speech = [
        make_speech(5.0, 8.0, " Hello everyone", 0.0),      # Whisper puts a space in front of every piece
        make_speech(20.0, 30.0, " today we start", 0.0),
        make_speech(105.0, 110.0, " second slide now", 100.0),
    ]

    objects = run_knowledge(tmp_path, slides, speech)

    assert len(objects) == 2
    assert objects[0]["transcript"] == "Hello everyone today we start"
    assert objects[1]["transcript"] == "second slide now"


def test_each_slide_ends_when_the_next_one_starts(tmp_path):
    slides = [make_slide(0.0), make_slide(100.0)]
    speech = [
        make_speech(5.0, 8.0, "a", 0.0),
        make_speech(105.0, 110.0, "b", 100.0),
    ]

    objects = run_knowledge(tmp_path, slides, speech)

    assert objects[0]["start_timestamp"] == 0.0
    assert objects[0]["end_timestamp"] == 100.0       # the next slide's start
    assert objects[1]["start_timestamp"] == 100.0
    # The last slide has no "next slide", so it ends when the last speech ends
    assert objects[1]["end_timestamp"] == 110.0


def test_a_slide_nobody_spoke_during_is_kept_with_an_empty_transcript(tmp_path):
    # A quick section-title slide: it must NOT be dropped, because its title is useful
    slides = [make_slide(0.0), make_slide(50.0), make_slide(100.0)]
    speech = [
        make_speech(5.0, 8.0, "first", 0.0),
        make_speech(105.0, 110.0, "third", 100.0),
    ]

    objects = run_knowledge(tmp_path, slides, speech)

    assert len(objects) == 3
    assert objects[1]["start_timestamp"] == 50.0
    assert objects[1]["transcript"] == ""


def test_speech_before_the_first_slide_is_not_put_on_any_slide(tmp_path):
    slides = [make_slide(10.0)]
    speech = [
        make_speech(1.0, 3.0, "before any slide", None),
        make_speech(12.0, 15.0, "during the slide", 10.0),
    ]

    objects = run_knowledge(tmp_path, slides, speech)

    assert len(objects) == 1
    assert objects[0]["transcript"] == "during the slide"


def test_no_spoken_word_is_lost_or_duplicated(tmp_path):
    # This is the same check we did by hand on the real lectures: total words must match
    slides = [make_slide(0.0), make_slide(50.0), make_slide(100.0)]
    speech = [
        make_speech(1.0, 2.0, " one two three", 0.0),
        make_speech(3.0, 4.0, " four five", 0.0),
        make_speech(60.0, 61.0, " six", 50.0),
        make_speech(110.0, 111.0, " seven eight nine ten", 100.0),
    ]

    objects = run_knowledge(tmp_path, slides, speech)

    words_in_objects = 0
    for knowledge_object in objects:
        words_in_objects = words_in_objects + len(knowledge_object["transcript"].split())

    assert words_in_objects == 10


def test_slides_listed_out_of_order_come_out_in_time_order(tmp_path):
    slides = [make_slide(100.0), make_slide(0.0)]
    speech = [make_speech(5.0, 8.0, "a", 0.0), make_speech(105.0, 110.0, "b", 100.0)]

    objects = run_knowledge(tmp_path, slides, speech)

    assert objects[0]["start_timestamp"] == 0.0
    assert objects[1]["start_timestamp"] == 100.0


def test_missing_vision_results_become_empty_values(tmp_path):
    # description None = "the vision stage failed for this slide"; content_type None = "not done"
    slides = [make_slide(0.0, description=None, content_type=None)]
    speech = [make_speech(5.0, 8.0, "a", 0.0)]

    objects = run_knowledge(tmp_path, slides, speech)

    assert objects[0]["slide_description"] == ""
    assert objects[0]["content_type"] == ""


def test_slide_information_is_carried_into_the_object(tmp_path):
    slides = [
        make_slide(
            0.0,
            description="A plot of two curves",
            content_type="chart",
            title="Sensitivity",
            slide_number=9,
        )
    ]
    speech = [make_speech(5.0, 8.0, "a", 0.0)]

    objects = run_knowledge(tmp_path, slides, speech)

    assert objects[0]["slide_text"] == "ocr 0.0"
    assert objects[0]["slide_description"] == "A plot of two curves"
    assert objects[0]["content_type"] == "chart"
    assert objects[0]["title"] == "Sensitivity"
    assert objects[0]["slide_number"] == 9
    assert objects[0]["image_path"] == "frame_0.0.jpg"


def test_the_lecture_id_is_the_name_of_the_output_folder(tmp_path):
    slides = [make_slide(0.0)]
    speech = [make_speech(5.0, 8.0, "a", 0.0)]

    objects = run_knowledge(tmp_path, slides, speech)

    assert objects[0]["lecture_id"] == "lecture_test"
