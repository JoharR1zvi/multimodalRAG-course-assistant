# For this piece of speech, which slide was being shown?

import bisect
import json
import sys
from pathlib import Path

from src.schemas.knowledge_object import (
    TranscriptSegment,
    VisualMetadata,
    AlignedSegment,
)


def align(
    transcript_path: Path,
    visual_metadata_path: Path,
    output_path: Path,
    *,
    force: bool = False,
) -> None:

    # Stop if the output already exists
    if output_path.exists():
        if not force:
            return

    # Read transcript (the file is closed once the "with" block ends)
    with open(transcript_path, encoding="utf-8") as f:
        transcript_data = json.load(f)

    segments = []
    for s in transcript_data:
        segment = TranscriptSegment(**s)
        segments.append(segment)

    # Read slide information
    with open(visual_metadata_path, encoding="utf-8") as f:
        visual_metadata_data = json.load(f)

    keyframes = []
    for k in visual_metadata_data:
        keyframe = VisualMetadata(**k)
        keyframes.append(keyframe)

    # Sort slides by timestamp
    def get_timestamp(keyframe):
        return keyframe.timestamp

    keyframes.sort(key=get_timestamp)

    # Make a list of just the timestamps
    keyframe_timestamps = []

    for keyframe in keyframes:
        keyframe_timestamps.append(keyframe.timestamp)

    # Store the aligned speech segments
    aligned = []

    # Go through each speech segment
    for segment in segments:

        # Find where the speech start time belongs
        insertion_position = bisect.bisect_right(
            keyframe_timestamps,
            segment.start,
        )

        # Get the slide before that position
        idx = insertion_position - 1

        # Get the slide timestamp
        if idx >= 0:
            keyframe_timestamp = keyframe_timestamps[idx]
        else:
            keyframe_timestamp = None

        # Create the aligned segment
        aligned_segment = AlignedSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            keyframe_timestamp=keyframe_timestamp,
        )

        # Add it to the list
        aligned.append(aligned_segment)

    # Convert everything to plain dictionaries BEFORE touching the output file,
    # so a failure here can't leave an empty/broken file behind
    output_data = []

    for aligned_segment in aligned:
        aligned_dictionary = aligned_segment.model_dump()
        output_data.append(aligned_dictionary)

    # Make sure the output folder exists
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write the result
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


# Run the program
if __name__ == "__main__":

    transcript_path = Path(sys.argv[1])
    visual_metadata_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    align(
        transcript_path,
        visual_metadata_path,
        output_path,
    )
