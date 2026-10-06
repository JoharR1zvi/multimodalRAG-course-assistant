# Tests for src/verify.py, the Phase 1 exit check.
#
# A checker is only worth trusting if it fails when it should. So each test here starts from a
# small, correct fake lecture, breaks exactly one thing, and checks that exactly that check fails.

import json

from src.schemas.knowledge_object import (
    AlignedSegment,
    KnowledgeObject,
    TranscriptSegment,
    VisualMetadata,
)
from src.verify import FAIL, WARN, check_lecture, check_lecture_data, describe_moment

LECTURE = "lecture_x"


def make_good_lecture():
    # A tiny, correct lecture: two slides (at 0 s and 100 s) and three pieces of speech.
    transcript = [
        TranscriptSegment(start=5.0, end=8.0, text=" hello there"),
        TranscriptSegment(start=20.0, end=30.0, text=" second piece here"),
        TranscriptSegment(start=105.0, end=110.0, text=" third slide words"),
    ]

    visual = []
    for timestamp in (0.0, 100.0):
        visual.append(
            VisualMetadata(
                timestamp=timestamp,
                image_path=f"frame_{timestamp:.2f}.jpg",
                text="ocr",
                description="",
                content_type="text_slide",
                title="A title",
                cleaned_text="ocr",
            )
        )

    knowledge = [
        KnowledgeObject(
            lecture_id=LECTURE, start_timestamp=0.0, end_timestamp=100.0,
            transcript="hello there second piece here",
            slide_text="ocr", slide_description="", image_path="frame_0.00.jpg",
        ),
        KnowledgeObject(
            lecture_id=LECTURE, start_timestamp=100.0, end_timestamp=110.0,
            transcript="third slide words",
            slide_text="ocr", slide_description="", image_path="frame_100.00.jpg",
        ),
    ]

    alignment = [
        AlignedSegment(start=5.0, end=8.0, text=" hello there", keyframe_timestamp=0.0),
        AlignedSegment(start=20.0, end=30.0, text=" second piece here", keyframe_timestamp=0.0),
        AlignedSegment(start=105.0, end=110.0, text=" third slide words", keyframe_timestamp=100.0),
    ]

    keyframe_filenames = ["frame_0.00.jpg", "frame_100.00.jpg"]

    return transcript, visual, knowledge, alignment, keyframe_filenames


def run_checks(transcript, visual, knowledge, alignment, keyframe_filenames, missing_images=None):
    if missing_images is None:
        missing_images = []

    return check_lecture_data(LECTURE, transcript, visual, knowledge, alignment, keyframe_filenames, missing_images)


def names_with_status(results, wanted_status):
    names = []
    for status, name, detail in results:
        if status == wanted_status:
            names.append(name)
    return names


def assert_only_this_check_fails(results, name_fragment):
    failed = names_with_status(results, FAIL)

    assert len(failed) == 1, f"expected exactly one failed check, got {failed}"
    assert name_fragment in failed[0]


# ---------- a correct lecture ----------

def test_a_correct_lecture_passes_every_check():
    results = run_checks(*make_good_lecture())

    assert names_with_status(results, FAIL) == []


# ---------- breaking one thing at a time ----------

def test_a_gap_between_two_slides_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    knowledge[0].end_timestamp = 99.0       # the first slide now ends one second before the next starts

    assert_only_this_check_fails(run_checks(transcript, visual, knowledge, alignment, files), "end times chain")


def test_lost_speech_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    knowledge[0].transcript = "hello there second here"      # the word "piece" disappeared

    assert_only_this_check_fails(run_checks(transcript, visual, knowledge, alignment, files), "no speech lost")


def test_duplicated_speech_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    knowledge[1].transcript = "third slide words words"       # one word twice

    assert_only_this_check_fails(run_checks(transcript, visual, knowledge, alignment, files), "no speech lost")


def test_a_missing_description_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    visual[1].description = None            # the vision step never finished this keyframe

    assert_only_this_check_fails(run_checks(transcript, visual, knowledge, alignment, files), "vision step complete")


def test_a_missing_cleaned_text_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    visual[0].cleaned_text = None

    assert_only_this_check_fails(run_checks(transcript, visual, knowledge, alignment, files), "OCR cleanup complete")


def test_a_missing_keyframe_file_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    files = ["frame_0.00.jpg"]               # one image file is gone

    assert_only_this_check_fails(run_checks(transcript, visual, knowledge, alignment, files), "keyframe files match")


def test_an_image_path_that_does_not_exist_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()

    results = run_checks(transcript, visual, knowledge, alignment, files, missing_images=["frame_0.00.jpg"])

    assert_only_this_check_fails(results, "every image path exists")


def test_the_wrong_lecture_id_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    knowledge[1].lecture_id = "some_other_lecture"

    assert_only_this_check_fails(run_checks(transcript, visual, knowledge, alignment, files), "names its lecture")


def test_one_slide_too_few_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    knowledge = knowledge[:1]

    failed = names_with_status(run_checks(transcript, visual, knowledge, alignment, files), FAIL)

    assert any("one knowledge object per keyframe" in name for name in failed)


def test_alignment_pointing_at_the_wrong_slide_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    alignment[2].keyframe_timestamp = 0.0    # the third piece of speech belongs to the 100 s slide

    assert_only_this_check_fails(run_checks(transcript, visual, knowledge, alignment, files), "alignment agrees")


def test_a_missing_aligned_segment_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    alignment = alignment[:2]

    failed = names_with_status(run_checks(transcript, visual, knowledge, alignment, files), FAIL)

    assert any("alignment covers every segment" in name for name in failed)


def test_an_empty_transcript_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()

    failed = names_with_status(run_checks([], visual, knowledge, [], files), FAIL)

    assert any("transcript has speech" in name for name in failed)


def test_a_transcript_out_of_time_order_is_caught():
    transcript, visual, knowledge, alignment, files = make_good_lecture()
    transcript[0], transcript[1] = transcript[1], transcript[0]

    failed = names_with_status(run_checks(transcript, visual, knowledge, alignment, files), FAIL)

    assert any("transcript is well formed" in name for name in failed)


# ---------- things that are warnings, not failures ----------

def test_a_slide_nobody_spoke_during_is_only_a_warning():
    transcript, visual, knowledge, alignment, files = make_good_lecture()

    # Add a quick third slide at 110 s with no speech, which ends when the lecture ends
    visual.append(VisualMetadata(
        timestamp=110.0, image_path="frame_110.00.jpg", text="", description="",
        content_type="title_slide", title="Next topic", cleaned_text="",
    ))
    knowledge.append(KnowledgeObject(
        lecture_id=LECTURE, start_timestamp=110.0, end_timestamp=110.0, transcript="",
        slide_text="", slide_description="", image_path="frame_110.00.jpg",
    ))
    files.append("frame_110.00.jpg")

    results = run_checks(transcript, visual, knowledge, alignment, files)

    assert names_with_status(results, FAIL) == []
    assert "slides nobody spoke during" in names_with_status(results, WARN)


def test_speech_before_the_first_slide_is_only_a_warning():
    transcript, visual, knowledge, alignment, files = make_good_lecture()

    # The first slide now appears at 4 s, and someone speaks at 1 s (before it).
    # The other two early pieces of speech (5 s and 20 s) are still after the first slide.
    visual[0].timestamp = 4.0
    visual[0].image_path = "frame_4.00.jpg"
    knowledge[0].start_timestamp = 4.0
    knowledge[0].image_path = "frame_4.00.jpg"
    files = ["frame_4.00.jpg", "frame_100.00.jpg"]

    transcript.insert(0, TranscriptSegment(start=1.0, end=3.0, text=" before any slide"))

    # Alignment must now say: 1 s -> no slide, 5 s -> the 4 s slide, 20 s -> the 4 s slide
    alignment[0].keyframe_timestamp = 4.0
    alignment[1].keyframe_timestamp = 4.0
    alignment.insert(0, AlignedSegment(start=1.0, end=3.0, text=" before any slide", keyframe_timestamp=None))

    results = run_checks(transcript, visual, knowledge, alignment, files)

    assert names_with_status(results, FAIL) == []
    assert "speech before the first keyframe" in names_with_status(results, WARN)


# ---------- the file loading part ----------

def write_json(path, items):
    dictionaries = []
    for item in items:
        dictionaries.append(item.model_dump())

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dictionaries, f)


def write_lecture_to_disk(processed_dir):
    # Writes the good fake lecture as real files, the way the pipeline would
    transcript, visual, knowledge, alignment, files = make_good_lecture()

    lecture_dir = processed_dir / LECTURE
    keyframes_dir = lecture_dir / "keyframes"
    keyframes_dir.mkdir(parents=True)

    # The image files must really exist, and visual_metadata.json must point at them
    for entry in visual:
        image_file = keyframes_dir / f"frame_{entry.timestamp:.2f}.jpg"
        image_file.write_bytes(b"not a real picture, but the file exists")
        entry.image_path = str(image_file)

    write_json(lecture_dir / "transcript.json", transcript)
    write_json(lecture_dir / "visual_metadata.json", visual)
    write_json(lecture_dir / "knowledge_objects.json", knowledge)
    write_json(lecture_dir / "alignment.json", alignment)

    return lecture_dir


def test_a_complete_lecture_folder_passes(tmp_path):
    write_lecture_to_disk(tmp_path)

    results = check_lecture(LECTURE, tmp_path)

    assert names_with_status(results, FAIL) == []


def test_a_missing_file_is_a_failure(tmp_path):
    lecture_dir = write_lecture_to_disk(tmp_path)
    (lecture_dir / "alignment.json").unlink()

    failed = names_with_status(check_lecture(LECTURE, tmp_path), FAIL)

    assert "alignment.json exists" in failed


def test_a_file_with_the_wrong_shape_is_a_failure(tmp_path):
    lecture_dir = write_lecture_to_disk(tmp_path)

    # A transcript segment must have start, end and text
    with open(lecture_dir / "transcript.json", "w", encoding="utf-8") as f:
        json.dump([{"start": 1.0}], f)

    failed = names_with_status(check_lecture(LECTURE, tmp_path), FAIL)

    assert "transcript.json loads and has the right shape" in failed


def test_a_missing_lecture_folder_is_a_failure(tmp_path):
    failed = names_with_status(check_lecture("no_such_lecture", tmp_path), FAIL)

    assert "lecture folder exists" in failed


def test_describe_moment_shows_the_slide_and_the_speech(tmp_path, capsys):
    write_lecture_to_disk(tmp_path)

    describe_moment(LECTURE, 25.0, tmp_path)

    printed = capsys.readouterr().out

    assert "slide shown from 0s to 100s" in printed      # the slide on screen at 25 s
    assert "second piece here" in printed                # the speech around 25 s
    assert "hello there" not in printed                  # 5 s is more than 10 s away


def test_describe_moment_outside_the_lecture_says_so(tmp_path, capsys):
    write_lecture_to_disk(tmp_path)

    describe_moment(LECTURE, 5000.0, tmp_path)

    assert "outside the lecture" in capsys.readouterr().out
