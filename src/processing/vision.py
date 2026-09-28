from google import genai
from PIL import Image
from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.schemas.knowledge_object import VisualMetadata
from pathlib import Path
import json
import time

client = genai.Client(api_key=GEMINI_API_KEY)

def describe_images(metadata_path: Path, *, force: bool = False) -> None:
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} doesn't exist - run OCR (step 8) first")

    with open(metadata_path, encoding="utf-8") as f:
        entries = [VisualMetadata(**e) for e in json.load(f)]

    pending = [e for e in entries if e.description is None or force]

    if not pending:
        print("Nothing to do - every entry already has a description.")
        return
    
    PROMPT = (
    "This image is a screenshot of a lecture slide being annotated in a screen-recording tool. "
    "Ignore the application menu/toolbar strip at the top of the screen and any small webcam video "
    "overlay in a corner - these are recording artifacts, not slide content. Describe only the "
    "diagram, chart, plot, or figure in the main slide area: what type of visual it is, its key "
    "elements (axes, labels, curves, arrows, regions, groups being compared), and what concept or "
    "relationship it communicates. If the slide contains only text/bullet points and no diagram, "
    "chart, plot, or figure, respond with exactly: 'No diagram present.'"
)
    
    
    for entry in pending:
        image = Image.open(entry.image_path)

        for attempt in range(4):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[PROMPT, image],
                )
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  rate limited or error ({e}), retrying in {wait}s...")
                time.sleep(wait)
        #The else runs if the for loop finishes normally without hitting break.
        else:
            print(f"  giving up on {entry.image_path} after 4 attempts")
            continue

        text = response.text.strip()
        entry.description = "" if text == "No diagram present." else text
        preview = entry.description[:60] if entry.description else "(no diagram)"
        print(f"[{entry.timestamp:.2f}s] {Path(entry.image_path).name}: {preview}")


    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump([e.model_dump() for e in entries], f, indent=2)



import sys
if __name__ == "__main__":
    metadata_path = Path(sys.argv[1])
    describe_images(metadata_path)
