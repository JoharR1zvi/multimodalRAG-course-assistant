# The Phase 1 exit check: does a processed lecture really have what Phase 1 promises?
#
# The promise: for ANY moment in the lecture you can tell what was being said and which
# slide was on screen. That only holds if the files are complete and agree with each other,
# so this script checks exactly that. It reads only the files in data/processed/<lecture>/.
# It needs no video, no GPU and no API key, and it never changes anything.
#
# Usage (from the project root):
#   python -m src.verify lecture_01              # check one lecture
#   python -m src.verify --all                   # check every folder in data/processed/
#   python -m src.verify lecture_01 --at 1420    # show what was said and shown at second 1420
#
# Every check ends as one of:
#   PASS  everything is as it should be
#   FAIL  something is broken (the exit code of the program becomes 1)
#   WARN  not broken, but worth a look
#   INFO  just a fact about the lecture

import argparse
import bisect
import json
import sys
from pathlib import Path

from src.config import CHUNK_MAX_WORDS
from src.schemas.chunk import Chunk
from src.schemas.knowledge_object import (
    AlignedSegment,
    KnowledgeObject,
    TranscriptSegment,
    VisualMetadata,
)

PROCESSED_DIR = Path("data/processed")

# How far (in seconds) the last speech piece of a chunk may run past the end of its slide
SPEECH_OVERRUN_SECONDS = 10.0

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
INFO = "INFO"


def add_result(results: list, status: str, name: str, detail: str = "") -> None:
    # Every check adds one line like ("PASS", "end times chain", "63 joins")
    results.append((status, name, detail))


def parse_keyframe_timestamp(filename: str) -> float:
    # "frame_180.00.jpg" -> 180.0
    stem = Path(filename).stem
    return float(stem.removeprefix("frame_"))


def count_words(text: str) -> int:
    return len(text.split())


def check_lecture_data(
    lecture_name: str,
    transcript: list,
    visual: list,
    knowledge: list,
    alignment: list,
    keyframe_filenames: list,
    missing_images: list,
) -> list:
    # Runs every check on data that is already loaded and validated.
    # Takes plain lists (no files), so it can be tested with tiny hand-made data.
    # transcript: list of TranscriptSegment      visual: list of VisualMetadata
    # knowledge:  list of KnowledgeObject        alignment: list of AlignedSegment
    # keyframe_filenames: names of the .jpg files in the keyframes folder
    # missing_images: image paths named in visual_metadata.json that do not exist on disk

    results = []

    # ---------- transcript ----------

    if len(transcript) == 0:
        add_result(results, FAIL, "transcript has speech", "transcript.json is empty")
    else:
        bad_times = 0
        out_of_order = 0

        for i in range(len(transcript)):
            if transcript[i].end < transcript[i].start:
                bad_times = bad_times + 1

            if i > 0 and transcript[i].start < transcript[i - 1].start:
                out_of_order = out_of_order + 1

        if bad_times == 0 and out_of_order == 0:
            add_result(results, PASS, "transcript is well formed", f"{len(transcript)} segments, in time order")
        else:
            add_result(results, FAIL, "transcript is well formed", f"{bad_times} segments end before they start, {out_of_order} out of order")

    # ---------- keyframes ----------

    visual_sorted = sorted(visual, key=lambda entry: entry.timestamp)

    if len(visual_sorted) == 0:
        add_result(results, FAIL, "keyframes exist", "visual_metadata.json is empty")
        # Nothing else can be checked without keyframes
        return results

    first_slide_time = visual_sorted[0].timestamp

    file_times = set()
    for filename in keyframe_filenames:
        file_times.add(round(parse_keyframe_timestamp(filename), 2))

    metadata_times = set()
    for entry in visual_sorted:
        metadata_times.add(round(entry.timestamp, 2))

    if len(keyframe_filenames) == len(visual_sorted) and file_times == metadata_times:
        add_result(results, PASS, "keyframe files match the metadata", f"{len(visual_sorted)} images, same timestamps")
    else:
        add_result(results, FAIL, "keyframe files match the metadata", f"{len(keyframe_filenames)} image files, {len(visual_sorted)} metadata entries")

    if len(missing_images) == 0:
        add_result(results, PASS, "every image path exists")
    else:
        add_result(results, FAIL, "every image path exists", f"{len(missing_images)} missing, first: {missing_images[0]}")

    duplicates = len(visual_sorted) - len(metadata_times)
    if duplicates == 0:
        add_result(results, PASS, "keyframe timestamps are unique")
    else:
        add_result(results, FAIL, "keyframe timestamps are unique", f"{duplicates} duplicates")

    # ---------- vision and cleanup are complete ----------

    no_description = 0
    no_content_type = 0
    no_cleaned_text = 0

    for entry in visual_sorted:
        if entry.description is None:
            no_description = no_description + 1
        if entry.content_type is None:
            no_content_type = no_content_type + 1
        if entry.cleaned_text is None:
            no_cleaned_text = no_cleaned_text + 1

    if no_description == 0 and no_content_type == 0:
        add_result(results, PASS, "vision step complete", "every keyframe has a description and a content type")
    else:
        add_result(results, FAIL, "vision step complete", f"{no_description} without description, {no_content_type} without content type (run the pipeline again)")

    if no_cleaned_text == 0:
        add_result(results, PASS, "OCR cleanup complete", "every keyframe has cleaned text")
    else:
        add_result(results, FAIL, "OCR cleanup complete", f"{no_cleaned_text} keyframes without cleaned text (run the pipeline again)")

    # ---------- knowledge objects ----------

    if len(knowledge) == len(visual_sorted):
        add_result(results, PASS, "one knowledge object per keyframe", f"{len(knowledge)} of {len(visual_sorted)}")
    else:
        add_result(results, FAIL, "one knowledge object per keyframe", f"{len(knowledge)} objects but {len(visual_sorted)} keyframes")

    wrong_lecture_id = 0
    for knowledge_object in knowledge:
        if knowledge_object.lecture_id != lecture_name:
            wrong_lecture_id = wrong_lecture_id + 1

    if wrong_lecture_id == 0:
        add_result(results, PASS, "every object names its lecture", f"lecture_id = {lecture_name}")
    else:
        add_result(results, FAIL, "every object names its lecture", f"{wrong_lecture_id} objects have a different lecture_id")

    # Slides must start exactly where the keyframes are
    start_mismatch = 0
    for i in range(min(len(knowledge), len(visual_sorted))):
        if knowledge[i].start_timestamp != visual_sorted[i].timestamp:
            start_mismatch = start_mismatch + 1

    if start_mismatch == 0:
        add_result(results, PASS, "objects start at their keyframe")
    else:
        add_result(results, FAIL, "objects start at their keyframe", f"{start_mismatch} objects start somewhere else")

    # Each slide must end exactly when the next one starts: no gaps and no overlaps
    broken_joins = 0
    for i in range(len(knowledge) - 1):
        if knowledge[i].end_timestamp != knowledge[i + 1].start_timestamp:
            broken_joins = broken_joins + 1

    backwards = 0
    for knowledge_object in knowledge:
        if knowledge_object.end_timestamp < knowledge_object.start_timestamp:
            backwards = backwards + 1

    if broken_joins == 0 and backwards == 0:
        add_result(results, PASS, "end times chain into the next start", f"{max(0, len(knowledge) - 1)} joins, no gaps, no overlaps")
    else:
        add_result(results, FAIL, "end times chain into the next start", f"{broken_joins} broken joins, {backwards} objects end before they start")

    # ---------- no speech lost ----------

    # Speech that starts before the first keyframe belongs to no slide, by design
    words_in_transcript = 0
    words_before_first_slide = 0

    for segment in transcript:
        if segment.start < first_slide_time:
            words_before_first_slide = words_before_first_slide + count_words(segment.text)
        else:
            words_in_transcript = words_in_transcript + count_words(segment.text)

    words_in_knowledge = 0
    for knowledge_object in knowledge:
        words_in_knowledge = words_in_knowledge + count_words(knowledge_object.transcript)

    if words_in_transcript == words_in_knowledge:
        add_result(results, PASS, "no speech lost or duplicated", f"{words_in_knowledge} words in the transcript = {words_in_knowledge} in the knowledge objects")
    else:
        add_result(results, FAIL, "no speech lost or duplicated", f"transcript has {words_in_transcript} words after the first slide, knowledge objects have {words_in_knowledge}")

    if words_before_first_slide > 0:
        add_result(results, WARN, "speech before the first keyframe", f"{words_before_first_slide} words belong to no slide")

    # ---------- the promise itself: any moment can be answered ----------

    # Every piece of speech must fall inside exactly one slide's time span,
    # and alignment.json must have picked that same slide.
    slide_starts = []
    for knowledge_object in knowledge:
        slide_starts.append(knowledge_object.start_timestamp)

    segments_outside = 0
    alignment_disagrees = 0

    if len(alignment) != len(transcript):
        add_result(results, FAIL, "alignment covers every segment", f"{len(alignment)} aligned segments but {len(transcript)} transcript segments")
    else:
        add_result(results, PASS, "alignment covers every segment", f"{len(alignment)} segments")

        for i in range(len(transcript)):
            segment = transcript[i]

            if segment.start < first_slide_time:
                expected_slide = None
            else:
                position = bisect.bisect_right(slide_starts, segment.start) - 1

                if position < 0 or position >= len(knowledge) or segment.start >= knowledge[position].end_timestamp:
                    segments_outside = segments_outside + 1
                    continue

                expected_slide = knowledge[position].start_timestamp

            if alignment[i].keyframe_timestamp != expected_slide:
                alignment_disagrees = alignment_disagrees + 1

    if segments_outside == 0:
        add_result(results, PASS, "every moment of speech is on some slide", "each segment falls inside a slide's time span")
    else:
        add_result(results, FAIL, "every moment of speech is on some slide", f"{segments_outside} segments fall outside every slide's time span")

    if alignment_disagrees == 0:
        add_result(results, PASS, "alignment agrees with the slide time spans")
    else:
        add_result(results, FAIL, "alignment agrees with the slide time spans", f"{alignment_disagrees} segments are assigned to a different slide")

    # ---------- things worth a look (never a failure) ----------

    blank_slides = []
    for knowledge_object in knowledge:
        if knowledge_object.transcript.strip() == "":
            blank_slides.append(int(knowledge_object.start_timestamp))

    if len(blank_slides) > 0:
        add_result(results, WARN, "slides nobody spoke during", f"{len(blank_slides)} slides, at seconds {blank_slides} (kept on purpose, they are often section titles)")

    no_title = 0
    for entry in visual_sorted:
        if entry.title == "":
            no_title = no_title + 1

    if no_title > 0:
        add_result(results, INFO, "keyframes without a title", f"{no_title} of {len(visual_sorted)}")

    # ---------- facts ----------

    type_counts = {}
    for entry in visual_sorted:
        content_type = str(entry.content_type)
        if content_type not in type_counts:
            type_counts[content_type] = 0
        type_counts[content_type] = type_counts[content_type] + 1

    add_result(results, INFO, "content types", str(type_counts))

    with_slide_number = 0
    for entry in visual_sorted:
        if entry.slide_number is not None:
            with_slide_number = with_slide_number + 1

    add_result(results, INFO, "slide numbers read from the screen", f"{with_slide_number} of {len(visual_sorted)}")

    if len(knowledge) > 0:
        span_minutes = (knowledge[-1].end_timestamp - knowledge[0].start_timestamp) / 60.0
        add_result(results, INFO, "time covered by the slides", f"{span_minutes:.1f} minutes")

    return results


def check_chunks_data(
    knowledge: list,
    alignment: list,
    chunks: list,
    max_words: int,
) -> list:
    # Phase 2 checks on chunks.json. Takes plain lists (no files), like check_lecture_data.
    # knowledge: list of KnowledgeObject      alignment: list of AlignedSegment
    # chunks: list of Chunk                   max_words: the hard maximum a chunk may have

    results = []

    if len(chunks) == 0:
        add_result(results, FAIL, "chunks exist", "chunks.json is empty")
        return results

    add_result(results, PASS, "chunks exist", f"{len(chunks)} chunks")

    # ---------- ids and neighbours ----------

    ids = set()
    for chunk in chunks:
        ids.add(chunk.chunk_id)

    if len(ids) == len(chunks):
        add_result(results, PASS, "chunk ids are unique")
    else:
        add_result(results, FAIL, "chunk ids are unique", f"{len(chunks) - len(ids)} duplicated ids")

    broken_links = 0
    for i in range(len(chunks)):
        if i > 0 and chunks[i].prev_chunk_id != chunks[i - 1].chunk_id:
            broken_links = broken_links + 1
        if i == 0 and chunks[i].prev_chunk_id is not None:
            broken_links = broken_links + 1
        if i < len(chunks) - 1 and chunks[i].next_chunk_id != chunks[i + 1].chunk_id:
            broken_links = broken_links + 1
        if i == len(chunks) - 1 and chunks[i].next_chunk_id is not None:
            broken_links = broken_links + 1

    if broken_links == 0:
        add_result(results, PASS, "chunks point at their neighbours")
    else:
        add_result(results, FAIL, "chunks point at their neighbours", f"{broken_links} broken prev/next links")

    # ---------- size ----------

    too_big = 0
    word_counts = []
    for chunk in chunks:
        words = count_words(chunk.text)
        word_counts.append(words)
        if words > max_words:
            too_big = too_big + 1

    if too_big == 0:
        add_result(results, PASS, "no chunk is over the maximum size", f"largest chunk has {max(word_counts)} words (maximum {max_words})")
    else:
        add_result(results, FAIL, "no chunk is over the maximum size", f"{too_big} chunks have more than {max_words} words")

    sorted_counts = sorted(word_counts)
    median_words = sorted_counts[len(sorted_counts) // 2]
    add_result(results, INFO, "chunk size in words", f"smallest {sorted_counts[0]}, median {median_words}, largest {sorted_counts[-1]}")

    # ---------- times lie inside the slides the chunk lists ----------

    slide_end = {}
    for knowledge_object in knowledge:
        slide_end[knowledge_object.start_timestamp] = knowledge_object.end_timestamp

    first_slide_time = min(slide_end.keys())

    outside = 0
    unknown_slide = 0
    for chunk in chunks:
        if len(chunk.slide_timestamps) == 0:
            outside = outside + 1
            continue

        known = True
        for slide_time in chunk.slide_timestamps:
            if slide_time not in slide_end:
                known = False
        if not known:
            unknown_slide = unknown_slide + 1
            continue

        earliest_start = min(chunk.slide_timestamps)
        latest_end = 0.0
        for slide_time in chunk.slide_timestamps:
            if slide_end[slide_time] > latest_end:
                latest_end = slide_end[slide_time]

        # Speech before the first slide is kept in the first chunk, so it may start before its slides
        starts_too_early = chunk.start_timestamp < earliest_start and chunk.start_timestamp >= first_slide_time

        # A speech piece belongs to the slide showing when it STARTS, so the last piece of a
        # chunk may run a few seconds past the end of that slide (7.4 s at most on our lectures)
        ends_too_late = chunk.end_timestamp > latest_end + SPEECH_OVERRUN_SECONDS

        if starts_too_early or ends_too_late:
            outside = outside + 1

    if outside == 0 and unknown_slide == 0:
        add_result(results, PASS, "chunk times lie inside their slides")
    else:
        add_result(results, FAIL, "chunk times lie inside their slides", f"{outside} chunks reach outside their slides, {unknown_slide} name an unknown slide")

    # ---------- nothing lost ----------

    # Every slide (blank section titles included) is listed by at least one chunk
    covered = set()
    for chunk in chunks:
        for slide_time in chunk.slide_timestamps:
            covered.add(slide_time)

    uncovered = 0
    for knowledge_object in knowledge:
        if knowledge_object.start_timestamp not in covered:
            uncovered = uncovered + 1

    if uncovered == 0:
        add_result(results, PASS, "every slide is in some chunk", f"{len(knowledge)} slides")
    else:
        add_result(results, FAIL, "every slide is in some chunk", f"{uncovered} slides are in no chunk")

    # Every speech piece appears, word for word, in a chunk that covers its time
    lost_pieces = 0
    for segment in alignment:
        found = False
        for chunk in chunks:
            if chunk.start_timestamp <= segment.start and segment.end <= chunk.end_timestamp:
                if segment.text.strip() in chunk.text:
                    found = True
                    break
        if not found:
            lost_pieces = lost_pieces + 1

    if lost_pieces == 0:
        add_result(results, PASS, "all speech is in some chunk", f"{len(alignment)} speech pieces found")
    else:
        add_result(results, FAIL, "all speech is in some chunk", f"{lost_pieces} speech pieces are in no chunk")

    return results


def load_json_list(path: Path, schema) -> list:
    # Reads a JSON file that holds a list, and checks every item against the schema
    with open(path, encoding="utf-8") as f:
        raw_items = json.load(f)

    items = []
    for raw_item in raw_items:
        items.append(schema(**raw_item))

    return items


def check_lecture(lecture_name: str, processed_dir: Path) -> list:
    # Loads one lecture's files and runs the checks. A missing or unreadable file is a FAIL.
    results = []
    lecture_dir = processed_dir / lecture_name

    if not lecture_dir.is_dir():
        add_result(results, FAIL, "lecture folder exists", f"{lecture_dir} not found")
        return results

    # The files Phase 1 promises, and what shape each one must have
    files_and_schemas = [
        ("transcript.json", TranscriptSegment),
        ("visual_metadata.json", VisualMetadata),
        ("knowledge_objects.json", KnowledgeObject),
        ("alignment.json", AlignedSegment),
    ]

    loaded = {}
    all_loaded = True

    for filename, schema in files_and_schemas:
        path = lecture_dir / filename

        if not path.exists():
            add_result(results, FAIL, f"{filename} exists", "file not found")
            all_loaded = False
            continue

        try:
            loaded[filename] = load_json_list(path, schema)
            add_result(results, PASS, f"{filename} loads and has the right shape", f"{len(loaded[filename])} items")
        except Exception as error:
            add_result(results, FAIL, f"{filename} loads and has the right shape", str(error)[:150])
            all_loaded = False

    keyframes_dir = lecture_dir / "keyframes"
    keyframe_filenames = []

    if keyframes_dir.is_dir():
        for file in keyframes_dir.iterdir():
            if file.suffix.lower() == ".jpg":
                keyframe_filenames.append(file.name)
    else:
        add_result(results, FAIL, "keyframes folder exists", f"{keyframes_dir} not found")
        all_loaded = False

    if not all_loaded:
        return results

    # Which image paths named in the metadata are not really there?
    missing_images = []
    for entry in loaded["visual_metadata.json"]:
        if not Path(entry.image_path).exists():
            missing_images.append(entry.image_path)

    results.extend(
        check_lecture_data(
            lecture_name,
            loaded["transcript.json"],
            loaded["visual_metadata.json"],
            loaded["knowledge_objects.json"],
            loaded["alignment.json"],
            keyframe_filenames,
            missing_images,
        )
    )

    # ---------- Phase 2: chunks.json (only checked when it exists) ----------

    chunks_path = lecture_dir / "chunks.json"

    if not chunks_path.exists():
        add_result(results, INFO, "chunks.json", "not built yet (run the pipeline to build it)")
        return results

    try:
        chunks = load_json_list(chunks_path, Chunk)
    except Exception as error:
        add_result(results, FAIL, "chunks.json loads and has the right shape", str(error)[:150])
        return results

    results.extend(
        check_chunks_data(
            loaded["knowledge_objects.json"],
            loaded["alignment.json"],
            chunks,
            CHUNK_MAX_WORDS,
        )
    )

    return results


def print_results(lecture_name: str, results: list) -> bool:
    # Prints one line per check. Returns True if nothing failed.
    print(f"\n===== {lecture_name} =====")

    failures = 0
    warnings = 0

    for status, name, detail in results:
        if status == FAIL:
            failures = failures + 1
        if status == WARN:
            warnings = warnings + 1

        if detail != "":
            print(f"  {status}  {name}: {detail}")
        else:
            print(f"  {status}  {name}")

    if failures == 0:
        print(f"  --> {lecture_name} PASSED ({warnings} warnings)")
    else:
        print(f"  --> {lecture_name} FAILED ({failures} failed checks)")

    return failures == 0


def describe_moment(lecture_name: str, seconds: float, processed_dir: Path) -> None:
    # Shows the Phase 1 promise in action: what was on screen and what was being said at one moment
    lecture_dir = processed_dir / lecture_name
    knowledge = load_json_list(lecture_dir / "knowledge_objects.json", KnowledgeObject)
    alignment = load_json_list(lecture_dir / "alignment.json", AlignedSegment)

    slide = None
    for knowledge_object in knowledge:
        if knowledge_object.start_timestamp <= seconds < knowledge_object.end_timestamp:
            slide = knowledge_object

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    print(f"\n{lecture_name} at {seconds:.0f}s ({minutes}:{remaining_seconds:02d})")

    if slide is None:
        print("  This moment is outside the lecture's slides.")
        return

    print(f"  On screen : slide shown from {slide.start_timestamp:.0f}s to {slide.end_timestamp:.0f}s")
    print(f"              type: {slide.content_type}   title: {slide.title!r}   slide number: {slide.slide_number}")
    print(f"              image: {slide.image_path}")

    if slide.slide_description != "":
        print(f"              description: {slide.slide_description[:200]}")

    print("  Being said around then (10 seconds either side):")

    printed = 0
    for segment in alignment:
        if segment.start >= seconds - 10 and segment.start <= seconds + 10:
            print(f"    [{segment.start:.0f}s] {segment.text.strip()}")
            printed = printed + 1

    if printed == 0:
        print("    (nobody speaks in that window)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 exit check for processed lectures.")
    parser.add_argument("lecture", nargs="?", help="folder name inside data/processed/, e.g. lecture_01")
    parser.add_argument("--all", action="store_true", help="check every lecture folder in data/processed/")
    parser.add_argument("--at", type=float, help="show what was on screen and what was said at this second")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="where the processed lectures are (default: data/processed)")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)

    if args.all and args.lecture:
        parser.error("give a lecture name OR --all, not both")

    if not args.all and not args.lecture:
        parser.error("give a lecture name, or use --all")

    if args.at is not None and not args.lecture:
        parser.error("--at needs a lecture name")

    if args.at is not None:
        describe_moment(args.lecture, args.at, processed_dir)
        return

    if args.all:
        lecture_names = []
        for folder in sorted(processed_dir.iterdir()):
            if folder.is_dir():
                lecture_names.append(folder.name)
    else:
        lecture_names = [args.lecture]

    passed = 0

    for lecture_name in lecture_names:
        results = check_lecture(lecture_name, processed_dir)

        if print_results(lecture_name, results):
            passed = passed + 1

    print(f"\n{passed} of {len(lecture_names)} lectures passed the Phase 1 exit check.")

    if passed != len(lecture_names):
        sys.exit(1)


if __name__ == "__main__":
    main()
