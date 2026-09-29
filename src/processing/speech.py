# Turns the lecture audio into text with timestamps, using Faster-Whisper.

import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel

from src.schemas.knowledge_object import TranscriptSegment


def transcribe(audio_path: Path, transcript_path: Path, *, force: bool = False) -> None:

    # Stop if the transcript already exists
    if transcript_path.exists():
        if not force:
            return

    # Load the model once (loading is the slow part). Runs on the GPU
    model = WhisperModel("small", device="cuda", compute_type="float16")

    # Whisper listens to the whole file and decides where each speech segment starts and ends.
    # "segments" is produced lazily: the real work happens while we loop over it below
    segments, info = model.transcribe(str(audio_path))

    print(f"Detected Language : {info.language}")

    results = []

    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        # Wrap in our schema so the shape is validated
        segment_obj = TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
        )
        results.append(segment_obj)

    # Convert to plain dictionaries before touching the output file
    output_data = []

    for segment_obj in results:
        output_data.append(segment_obj.model_dump())

    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


# Run the program
if __name__ == "__main__":

    audio_path = Path(sys.argv[1])
    transcript_path = Path(sys.argv[2])

    transcribe(audio_path, transcript_path)
