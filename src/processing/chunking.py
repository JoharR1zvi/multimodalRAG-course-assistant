# Cut a lecture into retrieval chunks: the pieces of text that get embedded and searched.
#
# Why not search slide by slide? Slides are very uneven: one holds 3 words of speech,
# the next 450. A chunk is a piece of about the same size (about 350 words of speech) that
# covers one stretch of the lecture, and it remembers exactly which slides and which
# seconds it covers.
#
# Input 1: alignment.json         -> every speech piece with its exact start/end time and its slide
# Input 2: knowledge_objects.json -> every slide: three versions of its text + diagram description
# Output : chunks.json            -> one chunk per stretch of the lecture
#
# How it works, in plain words:
#   1. Walk through the speech pieces in time order and drop them into a "bucket".
#   2. When the bucket holds about 350 words, seal it and start a new bucket.
#      The new bucket begins with a copy of the last piece of the sealed one (the overlap),
#      so an idea that got cut in half appears in both chunks.
#   3. A slide that nobody spoke during (often a section title) waits in a "waiting room"
#      and joins the NEXT bucket, because a title comes before the content it names.
#   4. A tiny last bucket is merged into the one before it.

import json
import sys
from pathlib import Path

from src.config import (
    COURSE_ID,
    CHUNK_TARGET_WORDS,
    CHUNK_MAX_WORDS,
    CHUNK_OVERLAP_SEGMENTS,
    CHUNK_MIN_LAST_WORDS,
    CHUNK_SLIDE_TEXT_SOURCE,
    CHUNK_INCLUDE_DESCRIPTION,
    CHUNK_SLIDE_AWARE,
    CHUNK_SLIDE_AWARE_MIN_WORDS,
)
from src.schemas.chunk import Chunk
from src.schemas.knowledge_object import AlignedSegment, KnowledgeObject

# The three versions of the slide text a chunk can embed, and "none" for no slide text at all
SLIDE_TEXT_SOURCES = ["slide_text", "cleaned_text", "clean_text", "none"]


def count_words(text: str) -> int:
    return len(text.split())


def new_bucket(carried_pieces: list) -> dict:
    # A bucket is a chunk that is still being filled.
    #   pieces         : list of (speech piece, time of its slide). The first ones may be the overlap.
    #   blank_times    : slides without speech that belong to this bucket
    #   new_count      : how many pieces are NEW (not copied over from the previous bucket)
    return {
        "pieces": list(carried_pieces),
        "blank_times": [],
        "new_count": 0,
    }


def bucket_words(bucket: dict) -> int:
    total = 0
    for segment, slide_time in bucket["pieces"]:
        total = total + count_words(segment.text)
    return total


def overlap_pieces(bucket: dict, overlap_segments: int) -> list:
    # The last few pieces of a sealed bucket, to be repeated at the start of the next one.
    # A bucket with no more pieces than the overlap is not repeated at all: the next bucket
    # must always move forward, never start where the previous one started.
    if overlap_segments <= 0:
        return []
    if len(bucket["pieces"]) <= overlap_segments:
        return []
    return bucket["pieces"][-overlap_segments:]


def is_real_slide_change(bucket: dict, next_slide_time, slide_by_time: dict) -> bool:
    # Is the next speech piece on a really different slide than the last piece of the bucket?
    # "Really different" means both slides have a title and the titles differ. A new screen state of
    # the same slide keeps its title, and a slide without a title gives no signal at all.
    if len(bucket["pieces"]) == 0:
        return False

    last_slide_time = bucket["pieces"][-1][1]

    if last_slide_time is None or next_slide_time is None:
        return False

    last_title = slide_by_time[last_slide_time].title.strip().lower()
    next_title = slide_by_time[next_slide_time].title.strip().lower()

    if last_title == "" or next_title == "":
        return False

    return last_title != next_title


def build_chunks(
    knowledge_objects: list,
    aligned_segments: list,
    *,
    course_id: str = COURSE_ID,
    target_words: int = CHUNK_TARGET_WORDS,
    max_words: int = CHUNK_MAX_WORDS,
    overlap_segments: int = CHUNK_OVERLAP_SEGMENTS,
    min_last_words: int = CHUNK_MIN_LAST_WORDS,
    slide_text_source: str = CHUNK_SLIDE_TEXT_SOURCE,
    include_description: bool = CHUNK_INCLUDE_DESCRIPTION,
    slide_aware: bool = CHUNK_SLIDE_AWARE,
    slide_aware_min_words: int = CHUNK_SLIDE_AWARE_MIN_WORDS,
) -> list:
    # knowledge_objects: list of KnowledgeObject      aligned_segments: list of AlignedSegment
    # Returns a list of Chunk, in time order.
    #   slide_text_source   : which version of the slide text is embedded ("none" = no slide text)
    #   include_description : whether the diagram/code description is embedded
    #   slide_aware         : seal a chunk at a real slide change once it has slide_aware_min_words
    #                         words, instead of at target_words (see is_real_slide_change)
    # Both only change the string that is embedded (embed_text). The chunk keeps every version of
    # the slide text and the description, so the answer step can still show them.

    if slide_text_source not in SLIDE_TEXT_SOURCES:
        raise ValueError(f"slide_text_source must be one of {SLIDE_TEXT_SOURCES}, not {slide_text_source!r}")

    if len(knowledge_objects) == 0 or len(aligned_segments) == 0:
        return []

    lecture_id = knowledge_objects[0].lecture_id

    # ---------- Step 1: slides in time order, and a way to find a slide by its start time ----------

    def get_start(knowledge_object):
        return knowledge_object.start_timestamp

    slides = sorted(knowledge_objects, key=get_start)

    slide_by_time = {}
    for slide in slides:
        slide_by_time[slide.start_timestamp] = slide

    # ---------- Step 2: sort the speech pieces into one list per slide ----------

    # Speech before the first slide belongs to no slide, but it is still speech worth searching
    leading_segments = []
    segments_by_slide = {}

    for segment in aligned_segments:
        slide_time = segment.keyframe_timestamp

        if slide_time is None:
            leading_segments.append(segment)
            continue

        # Speech that points at a slide that does not exist would silently vanish: refuse
        if slide_time not in slide_by_time:
            raise ValueError(f"A speech piece at {segment.start}s points at slide {slide_time}, which is not in the knowledge objects")

        if slide_time not in segments_by_slide:
            segments_by_slide[slide_time] = []
        segments_by_slide[slide_time].append(segment)

    # ---------- Step 3: one flat list of "things in time order" ----------

    # Each entry is (speech piece, slide time) for speech,
    # or (None, slide time) for a slide nobody spoke during
    ordered = []

    for segment in leading_segments:
        ordered.append((segment, None))

    for slide in slides:
        if slide.start_timestamp in segments_by_slide:
            for segment in segments_by_slide[slide.start_timestamp]:
                ordered.append((segment, slide.start_timestamp))
        else:
            ordered.append((None, slide.start_timestamp))

    # ---------- Step 4: fill the buckets ----------

    finished = []
    current = new_bucket([])
    waiting_room = []        # times of slides without speech, waiting for the next speech

    for segment, slide_time in ordered:

        # A slide nobody spoke during: wait for the next speech
        if segment is None:
            waiting_room.append(slide_time)
            continue

        words = count_words(segment.text)

        # 4a. Adding this piece would break the hard maximum: seal the bucket first
        if current["new_count"] > 0 and bucket_words(current) + words > max_words:
            finished.append(current)
            current = new_bucket(overlap_pieces(current, overlap_segments))

        # 4a'. Slide-aware cutting: the bucket is big enough and this piece starts a really new
        # slide, so this is a natural place to cut
        elif (
            slide_aware
            and current["new_count"] > 0
            and bucket_words(current) >= slide_aware_min_words
            and is_real_slide_change(current, slide_time, slide_by_time)
        ):
            finished.append(current)
            current = new_bucket(overlap_pieces(current, overlap_segments))

        # 4b. The waiting slides join the bucket this speech goes into
        for blank_time in waiting_room:
            current["blank_times"].append(blank_time)
        waiting_room = []

        # 4c. Put the piece into the bucket
        current["pieces"].append((segment, slide_time))
        current["new_count"] = current["new_count"] + 1

        # 4d. Big enough: seal it. With slide-aware cutting, a piece on a slide that has a title waits
        # for the next real slide change (or the hard maximum) instead; a slide without a title gives
        # no signal, so the length rule applies as before.
        if bucket_words(current) >= target_words:
            wait_for_slide_change = False

            if slide_aware and slide_time is not None:
                if slide_by_time[slide_time].title.strip() != "":
                    wait_for_slide_change = True

            if not wait_for_slide_change:
                finished.append(current)
                current = new_bucket(overlap_pieces(current, overlap_segments))

    # The bucket still open at the end counts only if it holds something new
    if current["new_count"] > 0:
        finished.append(current)

    # Slides without speech at the very end have no "next chunk", so they join the last one
    if len(waiting_room) > 0 and len(finished) > 0:
        for blank_time in waiting_room:
            finished[-1]["blank_times"].append(blank_time)

    # ---------- Step 5: merge a tiny last bucket into the one before it ----------

    if len(finished) >= 2:
        last = finished[-1]
        before = finished[-2]

        # The new pieces of the last bucket are its final new_count pieces
        first_new_index = len(last["pieces"]) - last["new_count"]
        new_pieces = last["pieces"][first_new_index:]

        new_words = 0
        for segment, slide_time in new_pieces:
            new_words = new_words + count_words(segment.text)

        if new_words < min_last_words and bucket_words(before) + new_words <= max_words:
            for piece in new_pieces:
                before["pieces"].append(piece)
            for blank_time in last["blank_times"]:
                before["blank_times"].append(blank_time)
            before["new_count"] = before["new_count"] + last["new_count"]
            finished.pop()

    # ---------- Step 6: turn every bucket into a Chunk ----------

    chunks = []
    used_ids = set()

    for bucket in finished:
        segments = []
        slide_times = set()

        for segment, slide_time in bucket["pieces"]:
            segments.append(segment)
            if slide_time is not None:
                slide_times.add(slide_time)

        for blank_time in bucket["blank_times"]:
            slide_times.add(blank_time)

        # What was said
        spoken_pieces = []
        for segment in segments:
            spoken_pieces.append(segment.text.strip())
        speech = " ".join(spoken_pieces)

        # What was on screen, slide by slide in time order
        image_paths = []
        slide_titles = []
        raw_texts = []
        cleaned_texts = []
        clean_texts = []
        descriptions = []

        for slide_time in sorted(slide_times):
            slide = slide_by_time[slide_time]

            image_paths.append(slide.image_path)

            if slide.title.strip() != "":
                slide_titles.append(slide.title.strip())
            if slide.slide_text.strip() != "":
                raw_texts.append(slide.slide_text.strip())
            if slide.cleaned_text.strip() != "":
                cleaned_texts.append(slide.cleaned_text.strip())
            if slide.clean_text.strip() != "":
                clean_texts.append(slide.clean_text.strip())
            if slide.slide_description.strip() != "":
                descriptions.append(slide.slide_description.strip())

        slide_text = "\n".join(raw_texts)
        cleaned_text = "\n".join(cleaned_texts)
        clean_text = "\n".join(clean_texts)
        slide_description = "\n".join(descriptions)

        # The string that gets embedded: slide text, then description, then speech
        if slide_text_source == "slide_text":
            chosen_slide_text = slide_text
        elif slide_text_source == "cleaned_text":
            chosen_slide_text = cleaned_text
        elif slide_text_source == "none":
            chosen_slide_text = ""
        else:
            chosen_slide_text = clean_text

        embed_parts = []
        if chosen_slide_text != "":
            embed_parts.append(chosen_slide_text)
        if include_description and slide_description != "":
            embed_parts.append(slide_description)
        embed_parts.append(speech)
        embed_text = "\n\n".join(embed_parts)

        # The chunk's times are the exact times of its speech
        start_timestamp = segments[0].start
        end_timestamp = segments[-1].end

        chunk_id = f"{lecture_id}_{int(start_timestamp)}"
        if chunk_id in used_ids:
            raise ValueError(f"Two chunks got the same id {chunk_id}")
        used_ids.add(chunk_id)

        chunk = Chunk(
            chunk_id=chunk_id,
            course_id=course_id,
            lecture_id=lecture_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            slide_timestamps=sorted(slide_times),
            image_paths=image_paths,
            slide_titles=slide_titles,
            text=speech,
            slide_text=slide_text,
            cleaned_text=cleaned_text,
            clean_text=clean_text,
            slide_description=slide_description,
            embed_text=embed_text,
        )
        chunks.append(chunk)

    # ---------- Step 7: link every chunk to its neighbours ----------

    for i in range(len(chunks)):
        if i > 0:
            chunks[i].prev_chunk_id = chunks[i - 1].chunk_id
        if i < len(chunks) - 1:
            chunks[i].next_chunk_id = chunks[i + 1].chunk_id

    return chunks


def chunk_lecture(
    alignment_path: Path,
    knowledge_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> None:
    # The pipeline stage: read the two input files, build the chunks, write chunks.json

    # Stop if the output already exists
    if output_path.exists():
        if not force:
            return

    with open(alignment_path, encoding="utf-8") as f:
        alignment_data = json.load(f)

    aligned_segments = []
    for a in alignment_data:
        aligned_segments.append(AlignedSegment(**a))

    with open(knowledge_path, encoding="utf-8") as f:
        knowledge_data = json.load(f)

    knowledge_objects = []
    for k in knowledge_data:
        knowledge_objects.append(KnowledgeObject(**k))

    # Build everything in memory first, so a failure never leaves a half-written file
    chunks = build_chunks(knowledge_objects, aligned_segments)

    output_data = []
    for chunk in chunks:
        output_data.append(chunk.model_dump())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Wrote {len(output_data)} chunks to {output_path}")


# Run the program
if __name__ == "__main__":

    alignment_path = Path(sys.argv[1])
    knowledge_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    chunk_lecture(
        alignment_path,
        knowledge_path,
        output_path,
    )
