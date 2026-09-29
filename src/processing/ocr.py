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


def extract_text(keyframes_dir: Path, output_path: Path, *, force: bool = False) -> None:

    # Stop if the output already exists
    if output_path.exists():
        if not force:
            return

    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        confidences = []

        for i in range(len(data["text"])):
            word = data["text"][i]
            confidence = float(data["conf"][i])

            # Skip empty "words" (Tesseract returns blanks for layout boxes)
            if word.strip() == "":
                continue

            words.append(word)

            # A confidence of -1 means "no confidence info", so ignore those
            if confidence >= 0:
                confidences.append(confidence)

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
        )
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
