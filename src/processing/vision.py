# Asks Gemini to describe the diagram/figure on each keyframe (OCR can't read diagrams).

import json
import sys
import time
from pathlib import Path

from google import genai
from PIL import Image

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.schemas.knowledge_object import VisualMetadata

client = genai.Client(api_key=GEMINI_API_KEY)

PROMPT = (
    "This image is a screenshot of a lecture slide being annotated in a screen-recording tool. "
    "Ignore the application menu/toolbar strip at the top of the screen and any small webcam video "
    "overlay in a corner - these are recording artifacts, not slide content. Describe only the "
    "diagram, chart, plot, or figure in the main slide area: what type of visual it is, its key "
    "elements (axes, labels, curves, arrows, regions, groups being compared), and what concept or "
    "relationship it communicates. If the slide contains only text/bullet points and no diagram, "
    "chart, plot, or figure, respond with exactly: 'No diagram present.'"
)

MAX_ATTEMPTS = 4


def describe_images(metadata_path: Path, *, force: bool = False) -> None:

    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} doesn't exist - run OCR (step 8) first")

    # Read the OCR results
    with open(metadata_path, encoding="utf-8") as f:
        metadata_data = json.load(f)

    entries = []
    for e in metadata_data:
        entry = VisualMetadata(**e)
        entries.append(entry)

    # Only describe entries that don't have a description yet (unless force is on)
    pending = []
    for entry in entries:
        if entry.description is None or force:
            pending.append(entry)

    if len(pending) == 0:
        print("Nothing to do - every entry already has a description.")
        return

    for entry in pending:
        image = Image.open(entry.image_path)

        # Try up to MAX_ATTEMPTS times, waiting longer after each failure (1s, 2s, 4s, 8s)
        response = None

        for attempt in range(MAX_ATTEMPTS):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[PROMPT, image],
                )
                break  # it worked, stop retrying
            except Exception as error:
                wait = 2 ** attempt
                print(f"  rate limited or error ({error}), retrying in {wait}s...")
                time.sleep(wait)

        # If every attempt failed, response is still None: skip this image and move on
        if response is None:
            print(f"  giving up on {entry.image_path} after {MAX_ATTEMPTS} attempts")
            continue

        text = response.text.strip()

        # "" means "this slide has no diagram"
        if text == "No diagram present.":
            entry.description = ""
        else:
            entry.description = text

        if entry.description:
            preview = entry.description[:60]
        else:
            preview = "(no diagram)"

        print(f"[{entry.timestamp:.2f}s] {Path(entry.image_path).name}: {preview}")

    # Convert to plain dictionaries before touching the file
    output_data = []

    for entry in entries:
        output_data.append(entry.model_dump())

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


# Run the program
if __name__ == "__main__":

    metadata_path = Path(sys.argv[1])

    describe_images(metadata_path)
