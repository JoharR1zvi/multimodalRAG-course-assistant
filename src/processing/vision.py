# Asks Gemini to analyse each keyframe: what kind of content it is, its title,
# a description of any diagram/code, the slide number, and the clean slide text.
# (OCR can't read diagrams, and it mangles code indentation.)
#
# Gemini fills in a fixed form (the SlideAnalysis schema) instead of writing free text,
# so we never have to guess what a sentence means.
#
# Speed: most of the time is spent WAITING for Gemini to answer. So several keyframes are
# sent at the same time, each in its own "worker thread" (see GEMINI_MAX_WORKERS in config.py).

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GEMINI_MAX_WORKERS,
    GEMINI_REQUESTS_PER_MINUTE,
)
from src.schemas.knowledge_object import SlideAnalysis, VisualMetadata

client = genai.Client(api_key=GEMINI_API_KEY)

# Generic on purpose: no mention of any specific recording software or layout.
PROMPT = (
    "This image is one frame from a recorded lecture video. It may show a presentation slide, "
    "a code editor or terminal, a document, or other teaching material.\n"
    "\n"
    "Ignore everything that is not teaching content: application menus, toolbars, window borders, "
    "and any small camera video of the presenter. If the presentation program shows a page or "
    "slide counter somewhere on screen, you may read it for slide_number.\n"
    "\n"
    "Fill in the form like this:\n"
    "- title: the slide's title exactly as shown. Empty string if there is no title.\n"
    "- description: ONLY for a figure that carries teaching content. If the frame contains a "
    "diagram, plot, chart, drawing, or picture of a table, describe it thoroughly in 3 to 6 "
    "sentences: what kind of visual it is, its key elements (axes, labels, curves, arrows, "
    "regions, groups being compared) and what concept or relationship it communicates. Name "
    "every label, axis title, legend entry, annotation, caption and marked value that you can "
    "read in the figure (leave out colors and line styles unless they carry meaning). If the "
    "frame shows code, write out the visible code exactly, keeping its indentation, then add one "
    "sentence saying what it does. Small decorative items (logos, icons, colored badges, "
    "bullet markers) are NOT figures. Mathematical formulas and all ordinary text belong in "
    "clean_text, not here. If there is no figure and no code, use an empty string: do NOT "
    "summarize or repeat the slide's text.\n"
    "- content_type: choose this AFTER writing the description. If the description covers a "
    "figure, diagram, plot or chart, pick diagram or chart, even when the slide also has text. "
    "Pick equation if the slide is mainly mathematical formulas. Use text_slide when the frame "
    "has only text (bullets, definitions, a formula inside a sentence) and no figure. If the "
    "frame is a camera shot of the presenter, a room or an audience and shows no teaching "
    "content on screen, pick other.\n"
    "- slide_number: the slide/page number if you can see one, otherwise null.\n"
    "- clean_text: all the teaching text visible on screen (including any clearly legible "
    "handwritten notes), without menus, toolbars or other interface text. Empty string if none."
)

# temperature 0 = the model picks its most likely answer, so reruns vary as little as possible
TEMPERATURE = 0.0

# How hard the model "thinks" before answering. None = use the model's default.
# Lower thinking = faster. Allowed: "MINIMAL", "LOW", "MEDIUM", "HIGH".
THINKING_LEVEL = "MINIMAL"

MAX_ATTEMPTS = 4

# ---------- Rate limiting ----------
# Gemini's free tier only accepts a fixed number of requests per minute. If we send faster,
# it answers "429 quota exceeded". So every worker thread must first take a numbered "slot",
# and the slots are spaced evenly in time (like a ticket queue with one ticket per ~4 seconds).
SECONDS_BETWEEN_REQUESTS = 60.0 / GEMINI_REQUESTS_PER_MINUTE

_slot_lock = threading.Lock()          # makes sure two threads never take the same slot
_next_free_slot_time = 0.0             # the earliest moment the next request may be sent


def wait_for_request_slot() -> None:
    global _next_free_slot_time

    # Taking a slot must be done by one thread at a time, so we hold the lock briefly
    with _slot_lock:
        now = time.time()

        # Our slot is either "now" (if nobody is queued) or the next free slot
        if _next_free_slot_time > now:
            my_slot_time = _next_free_slot_time
        else:
            my_slot_time = now

        # The next thread's slot comes one interval after ours
        _next_free_slot_time = my_slot_time + SECONDS_BETWEEN_REQUESTS

    # Sleep (outside the lock, so other threads can still take their slots) until our slot
    seconds_to_wait = my_slot_time - time.time()

    if seconds_to_wait > 0:
        time.sleep(seconds_to_wait)


def build_gemini_config() -> types.GenerateContentConfig:
    # Settings sent with every request: "answer in JSON shaped like SlideAnalysis"
    thinking_config = None

    if THINKING_LEVEL is not None:
        thinking_config = types.ThinkingConfig(thinking_level=THINKING_LEVEL)

    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=SlideAnalysis,
        temperature=TEMPERATURE,
        thinking_config=thinking_config,
    )


def explain_bad_reply(response) -> str:
    # Called when Gemini answered but the answer is not a valid SlideAnalysis.
    # Collects WHY (finish reason + start of the reply) so failures can be diagnosed.
    finish_reason = "unknown"

    if response.candidates:
        finish_reason = str(response.candidates[0].finish_reason)

    reply_text = response.text

    if reply_text is None:
        reply_start = "(no text in the reply)"
    else:
        reply_start = reply_text[:150].replace("\n", " ")

    return f"finish_reason={finish_reason}, reply starts: {reply_start!r}"


def analyse_one_image(entry: VisualMetadata, gemini_config: types.GenerateContentConfig):
    # Sends ONE keyframe to Gemini and returns a SlideAnalysis, or None if every attempt failed.
    # This function runs inside a worker thread, several at the same time.
    # It must not change shared data: it only returns its result, and the main thread stores it.
    name = Path(entry.image_path).name

    # Try up to MAX_ATTEMPTS times, waiting longer after each failure (1s, 2s, 4s, 8s)
    for attempt in range(MAX_ATTEMPTS):
        try:
            image = Image.open(entry.image_path)

            # Wait for our turn so we never exceed the requests-per-minute quota
            # (retries also take a slot, because a failed request still counts)
            wait_for_request_slot()

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[PROMPT, image],
                config=gemini_config,
            )

            # response.parsed is the reply already turned into a SlideAnalysis object.
            # It is None if the reply did not match the form - count that as a failure.
            analysis = response.parsed

            if analysis is None:
                raise ValueError("the reply did not match the form (" + explain_bad_reply(response) + ")")

            return analysis

        except Exception as error:
            wait = 2 ** attempt
            print(f"  {name}: error ({error}), retrying in {wait}s...")
            time.sleep(wait)

    print(f"  giving up on {entry.image_path} after {MAX_ATTEMPTS} attempts")
    return None


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

    gemini_config = build_gemini_config()

    # Start the stopwatch so we can report how long the whole stage took
    start_time = time.time()
    finished_count = 0
    interrupted = False

    # A "pool" of worker threads. We hand it one job per keyframe; it runs
    # GEMINI_MAX_WORKERS of them at the same time and starts the next job whenever one finishes.
    with ThreadPoolExecutor(max_workers=GEMINI_MAX_WORKERS) as pool:

        # job -> the entry it belongs to (so we know where to put each result)
        jobs = {}

        for entry in pending:
            job = pool.submit(analyse_one_image, entry, gemini_config)
            jobs[job] = entry

        try:
            # as_completed hands back each job the moment it finishes (not in the original order)
            for job in as_completed(jobs):
                entry = jobs[job]
                analysis = job.result()
                finished_count = finished_count + 1

                # None = every attempt failed. The entry's description stays None,
                # so the next run will try this keyframe again.
                if analysis is None:
                    continue

                # Copy every box of the form into the entry (only the main thread does this).
                # description "" means "nothing to describe", None would mean "not done yet".
                entry.content_type = analysis.content_type
                entry.title = analysis.title.strip()
                entry.description = analysis.description.strip()
                entry.slide_number = analysis.slide_number
                entry.clean_text = analysis.clean_text.strip()

                if entry.description:
                    preview = entry.description[:60]
                else:
                    preview = "(nothing to describe)"

                name = Path(entry.image_path).name
                print(f"[{finished_count}/{len(pending)}] {name}: {entry.content_type} | {preview}")

        except KeyboardInterrupt:
            # Ctrl+C: stop the jobs that have not started, but still save the finished ones below
            print("\nInterrupted - saving the keyframes that are already done...")
            pool.shutdown(wait=False, cancel_futures=True)
            interrupted = True

    elapsed = time.time() - start_time
    print(f"Processed {finished_count} of {len(pending)} keyframes in {elapsed:.1f}s")

    # Convert to plain dictionaries before touching the file
    output_data = []

    for entry in entries:
        output_data.append(entry.model_dump())

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Now that the results are saved, let Ctrl+C stop the whole program as usual
    if interrupted:
        raise KeyboardInterrupt


# Run the program
if __name__ == "__main__":

    metadata_path = Path(sys.argv[1])

    describe_images(metadata_path)
