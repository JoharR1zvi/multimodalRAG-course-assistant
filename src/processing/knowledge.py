# Merge slide info + speech into one "Knowledge Object" per slide (keyframe).
#
# Input 1: alignment.json        -> every speech segment, tagged with its slide's timestamp
# Input 2: visual_metadata.json  -> every slide: OCR text + Gemini description
# Output : knowledge_objects.json -> one object per slide:
#          "what was on the slide" + "everything said while it was showing"

import json
import sys
from pathlib import Path

from src.schemas.knowledge_object import (
    AlignedSegment,
    VisualMetadata,
    KnowledgeObject,
)


def build_knowledge_objects(
    alignment_path: Path,
    visual_metadata_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> None:

    # Stop if the output already exists
    if output_path.exists():
        if not force:
            return

    # The lecture id is just the name of the folder the output goes in
    # e.g. data/processed/lecture_01/knowledge_objects.json -> "lecture_01"
    lecture_id = output_path.parent.name

    # ---------- Step 1: read both input files ----------

    with open(alignment_path, encoding="utf-8") as f:
        alignment_data = json.load(f)

    aligned_segments = []
    for a in alignment_data:
        aligned_segment = AlignedSegment(**a)
        aligned_segments.append(aligned_segment)

    with open(visual_metadata_path, encoding="utf-8") as f:
        visual_metadata_data = json.load(f)

    keyframes = []
    for k in visual_metadata_data:
        keyframe = VisualMetadata(**k)
        keyframes.append(keyframe)

    # Slides must be in time order so "the next slide" really is the next one
    def get_timestamp(keyframe):
        return keyframe.timestamp

    keyframes.sort(key=get_timestamp)

    # The very last slide has no "next slide" to tell us when it ends,
    # so we use the end time of the last speech segment instead
    last_segment_end = aligned_segments[-1].end

    # ---------- Step 2: sort the speech into "buckets", one bucket per slide ----------

    # key = slide timestamp, value = list of speech texts spoken during that slide
    speech_by_slide = {}

    for segment in aligned_segments:
        slide_time = segment.keyframe_timestamp

        # Speech before the first slide has no slide, so skip it
        if slide_time is None:
            continue

        # First piece of speech for this slide: make a new empty bucket
        if slide_time not in speech_by_slide:
            speech_by_slide[slide_time] = []

        # .strip() removes the stray leading space Whisper puts on every piece
        speech_by_slide[slide_time].append(segment.text.strip())

    # ---------- Step 3: build one knowledge object per slide ----------

    knowledge_objects = []

    # enumerate gives us the position (i) AND the slide, so we can look at keyframes[i + 1]
    for i, keyframe in enumerate(keyframes):

        # 3a. Look up the bucket for this slide (a slide nobody spoke during gets an empty list)
        if keyframe.timestamp in speech_by_slide:
            spoken_pieces = speech_by_slide[keyframe.timestamp]
        else:
            spoken_pieces = []

        # Join the pieces into one paragraph, separated by spaces
        transcript = " ".join(spoken_pieces)

        # 3b. Work out when this slide stopped being shown
        is_last_slide = (i == len(keyframes) - 1)

        if is_last_slide:
            end_timestamp = last_segment_end
        else:
            next_keyframe = keyframes[i + 1]
            end_timestamp = next_keyframe.timestamp

        # 3c. description can be None (if Gemini failed) - the schema needs a string, so use "" instead
        if keyframe.description is None:
            slide_description = ""
        else:
            slide_description = keyframe.description

        # 3d. content_type can be None (vision not done for this slide) - the schema needs a string
        if keyframe.content_type is None:
            content_type = ""
        else:
            content_type = keyframe.content_type

        # 3e. cleaned_text can be None (OCR cleanup not done) - the schema needs a string
        # (clean_text is already a plain string with "" as its default)
        if keyframe.cleaned_text is None:
            cleaned_text = ""
        else:
            cleaned_text = keyframe.cleaned_text

        # 3f. Build the object (pydantic checks every field has the right type)
        knowledge_object = KnowledgeObject(
            lecture_id=lecture_id,
            start_timestamp=keyframe.timestamp,
            end_timestamp=end_timestamp,
            transcript=transcript,
            slide_text=keyframe.text,
            slide_description=slide_description,
            image_path=keyframe.image_path,
            content_type=content_type,
            title=keyframe.title,
            slide_number=keyframe.slide_number,
            cleaned_text=cleaned_text,
            clean_text=keyframe.clean_text,
        )

        knowledge_objects.append(knowledge_object)

    # ---------- Step 4: write the result ----------

    # Convert to plain dictionaries BEFORE opening the output file
    output_data = []

    for knowledge_object in knowledge_objects:
        output_data.append(knowledge_object.model_dump())

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"Wrote {len(output_data)} knowledge objects to {output_path}")


# Run the program
if __name__ == "__main__":

    alignment_path = Path(sys.argv[1])
    visual_metadata_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    build_knowledge_objects(
        alignment_path,
        visual_metadata_path,
        output_path,
    )
