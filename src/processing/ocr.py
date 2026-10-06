# Reads the text off each keyframe image using Tesseract (OCR).

import json
import sys
from pathlib import Path

import pytesseract
from PIL import Image

from src.config import TESSERACT_CMD
from src.schemas.knowledge_object import VisualMetadata

# If .env gives a path to tesseract.exe, tell pytesseract where it is
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _parse_timestamp(image_path: Path) -> float:
    # "frame_180.00.jpg" -> stem is "frame_180.00" -> remove "frame_" -> 180.0
    stem = image_path.stem
    number_text = stem.removeprefix("frame_")
    return float(number_text)


# Fields written by the vision stage (vision.py). When OCR is redone, these are copied over
# from the old file, so redoing OCR does NOT throw away the paid Gemini results.
VISION_FIELDS = ["description", "content_type", "title", "slide_number", "clean_text"]


def extract_text(keyframes_dir: Path, output_path: Path, *, force: bool = False) -> None:

    # Stop if the output already exists
    if output_path.exists():
        if not force:
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If we are REDOING OCR, remember what the vision stage already stored, keyed by timestamp
    old_entries_by_timestamp = {}

    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            old_data = json.load(f)

        for old_entry in old_data:
            old_entries_by_timestamp[old_entry["timestamp"]] = old_entry

    # Find all the keyframe images and sort them by timestamp
    image_paths = list(keyframes_dir.glob("*.jpg"))
    image_paths.sort(key=_parse_timestamp)

    results = []

    for image_path in image_paths:
        timestamp = _parse_timestamp(image_path)
        image = Image.open(image_path)

        # Tesseract returns a dictionary of parallel lists: one entry per detected word
        # data["text"][i] is the word, data["conf"][i] is how sure Tesseract is about it
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        words = []
        confidences = []              # only the real numbers, used for the average
        word_confidences = []         # one number per word, same order as `words`, 0 to 1

        for i in range(len(data["text"])):
            word = data["text"][i]
            confidence = float(data["conf"][i])

            # Skip empty "words" (Tesseract returns blanks for layout boxes)
            if word.strip() == "":
                continue

            words.append(word)

            # A confidence of -1 means "no confidence info", so ignore those in the average
            if confidence >= 0:
                confidences.append(confidence)
                word_confidences.append(confidence / 100.0)
            else:
                word_confidences.append(-1.0)

        # Average confidence, scaled from 0-100 to 0-1. None if we have no numbers
        if len(confidences) > 0:
            average_confidence = sum(confidences) / len(confidences) / 100.0
        else:
            average_confidence = None

        metadata = VisualMetadata(
            timestamp=timestamp,
            image_path=str(image_path),
            text="\n".join(words),
            confidence=average_confidence,
            word_confidences=word_confidences,
        )

        # Copy the vision results over from the old file, if this keyframe was already described.
        # (cleaned_text is NOT copied: the OCR text may have changed, so it must be cleaned again.)
        if timestamp in old_entries_by_timestamp:
            old_entry = old_entries_by_timestamp[timestamp]

            for field_name in VISION_FIELDS:
                if field_name in old_entry:
                    setattr(metadata, field_name, old_entry[field_name])

        results.append(metadata)

        print(f"[{timestamp:.2f}s] {image_path.name}: {len(words)} lines, confidence={average_confidence}")

    # Convert to plain dictionaries before touching the output file
    output_data = []

    for metadata in results:
        output_data.append(metadata.model_dump())

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


# Run the program
if __name__ == "__main__":

    keyframes_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    extract_text(keyframes_dir, output_path)
