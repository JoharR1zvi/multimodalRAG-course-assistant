# Cleans the raw OCR text of every keyframe. Two layers, because they catch different junk:
#
#   Layer 1: INTERFACE WORDS. Menu words like "File Edit View Page Tools" are read by OCR with
#            HIGH confidence (they are real words!), so a confidence filter can't see them.
#            Instead we count how many keyframes each word appears on. Words on many
#            keyframes are candidates. We ask Gemini ONCE per lecture which candidates are
#            software interface text, and save the answer in junk_terms.json (so you can
#            read it). Then plain code removes those words.
#
#   Layer 2: LOW CONFIDENCE. Garbage like "oP" or "sso" (OCR misreading handwriting or
#            icons) comes with a LOW confidence score, so we drop words below a cutoff.
#            Symbols with no letters or digits at all ("|", ">") are dropped too.
#
# The raw OCR text is never changed. The result goes into a NEW field, cleaned_text,
# so raw OCR, cleaned OCR and Gemini's own text can be compared fairly later.
#
# Input and output: visual_metadata.json (edited in place, like the vision stage)
# Extra output   : junk_terms.json (the inspectable list of candidates and Gemini's answer)

import json
import re
import sys
import time
from pathlib import Path

from google.genai import types

from src.config import GEMINI_MODEL
from src.processing.vision import client, wait_for_request_slot
from src.schemas.knowledge_object import JunkTermsAnswer, VisualMetadata

MAX_ATTEMPTS = 4

# How many example OCR texts to show Gemini, so it can see what the interface text looks like
NUMBER_OF_SAMPLE_FRAMES = 3
SAMPLE_CHARACTERS = 300


def normalize_word(word: str) -> str:
    # "File."  ->  "file"      "(Options)"  ->  "options"
    # Lowercase, then remove anything that is not a letter or digit from both ends of the word.
    lowered = word.lower()
    return re.sub(r"^\W+|\W+$", "", lowered)


def has_letter_or_digit(word: str) -> bool:
    # "|" and ">" and "®" have none, "hello.py" and "4096" do
    return re.search(r"\w", word) is not None


def get_raw_words(entry: VisualMetadata) -> list:
    # The raw OCR text has one word per line
    if entry.text == "":
        return []

    return entry.text.split("\n")


def find_candidates(entries: list, settings: dict) -> list:
    # Returns [(word, number_of_keyframes_it_appears_on), ...], most frequent first.
    # Only words that appear on many keyframes can be interface text.
    slides_per_word = {}

    for entry in entries:
        words_on_this_keyframe = set()

        for raw_word in get_raw_words(entry):
            word = normalize_word(raw_word)

            # Interface words are plain letters ("file", "tools"), not numbers or code
            if word.isalpha() and len(word) >= 3:
                words_on_this_keyframe.add(word)

        for word in words_on_this_keyframe:
            if word not in slides_per_word:
                slides_per_word[word] = 0
            slides_per_word[word] = slides_per_word[word] + 1

    # A word must appear on at least this many keyframes to be a candidate
    needed_by_share = settings["junk_min_slide_share"] * len(entries)
    needed = max(settings["junk_min_slides"], needed_by_share)

    candidates = []

    for word in slides_per_word:
        if slides_per_word[word] >= needed:
            candidates.append((word, slides_per_word[word]))

    # Most frequent first, then keep only the top few
    candidates.sort(key=lambda pair: pair[1], reverse=True)

    return candidates[: settings["junk_max_candidates"]]


def build_prompt(candidates: list, entries: list) -> str:
    # The question we send to Gemini: the candidate words plus a few example OCR texts

    lines = []
    lines.append("Text recognition (OCR) was run on screenshots taken from ONE lecture video.")
    lines.append("The words below each appear on many of those screenshots.")
    lines.append("")
    lines.append("Some of them are SOFTWARE INTERFACE text: menu names, toolbar labels, tooltips,")
    lines.append("window title bars, or page counters of the recording or presentation program.")
    lines.append("Others are ordinary words, or part of the lecture content itself (slide text,")
    lines.append("code, or terminal output the presenter is showing). Content is NOT junk.")
    lines.append("")
    lines.append("Return only the interface words. If you are unsure about a word, leave it out.")
    lines.append("")
    lines.append(f"Candidate words, with the number of screenshots (out of {len(entries)}) each appears on:")

    for word, count in candidates:
        lines.append(f"{word}: {count}")

    lines.append("")
    lines.append("Example OCR output of a few screenshots from the same video (words separated by spaces):")

    # Pick a few screenshots spread evenly across the lecture
    step = max(1, len(entries) // NUMBER_OF_SAMPLE_FRAMES)

    for number in range(NUMBER_OF_SAMPLE_FRAMES):
        position = min(number * step, len(entries) - 1)
        sample_words = get_raw_words(entries[position])
        sample_text = " ".join(sample_words)[:SAMPLE_CHARACTERS]
        lines.append(f"- {sample_text}")

    return "\n".join(lines)


def ask_gemini_for_interface_words(candidates: list, entries: list) -> list:
    # One request: "which of these frequent words are interface text?" Returns a list of words.
    prompt = build_prompt(candidates, entries)

    gemini_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=JunkTermsAnswer,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
    )

    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            # Same shared rate limiter as the vision stage (free tier: 15 requests per minute)
            wait_for_request_slot()

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt],
                config=gemini_config,
            )

            answer = response.parsed

            if answer is None:
                raise ValueError("the reply did not match the JunkTermsAnswer form")

            # Gemini may only pick words from the candidate list (never invent new ones)
            allowed = set()
            for word, count in candidates:
                allowed.add(word)

            interface_words = []

            for word in answer.interface_words:
                normalized = normalize_word(word)
                if normalized in allowed and normalized not in interface_words:
                    interface_words.append(normalized)

            return interface_words

        except Exception as error:
            last_error = error
            wait = 2 ** attempt
            print(f"  error ({error}), retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Could not get the interface-word list from Gemini after {MAX_ATTEMPTS} attempts: {last_error}")


def load_saved_junk_words(junk_terms_path: Path, number_of_entries: int):
    # Returns the junk word list from an earlier run, or None if there is no usable file.
    # "Usable" = it was made for the same number of keyframes (otherwise the keyframes changed).
    if not junk_terms_path.exists():
        return None

    with open(junk_terms_path, encoding="utf-8") as f:
        saved = json.load(f)

    if saved.get("keyframes_counted") != number_of_entries:
        return None

    return saved["interface_words"]


def clean_one_entry(entry: VisualMetadata, junk_words: set, min_confidence: float, counters: dict) -> str:
    # Returns the cleaned text of one keyframe, and adds to the counters how many words
    # were dropped for which reason.
    raw_words = get_raw_words(entry)

    # The per-word confidences must line up one-to-one with the words
    has_confidences = len(entry.word_confidences) == len(raw_words)

    if not has_confidences and len(raw_words) > 0:
        counters["keyframes_without_confidences"] = counters["keyframes_without_confidences"] + 1

    kept_words = []

    for position, raw_word in enumerate(raw_words):
        counters["words_before"] = counters["words_before"] + 1

        if not has_letter_or_digit(raw_word):
            counters["dropped_symbol"] = counters["dropped_symbol"] + 1
            continue

        if normalize_word(raw_word) in junk_words:
            counters["dropped_interface"] = counters["dropped_interface"] + 1
            continue

        if has_confidences:
            confidence = entry.word_confidences[position]

            # -1 means "Tesseract gave no number", so we keep those words
            if confidence >= 0 and confidence < min_confidence:
                counters["dropped_low_confidence"] = counters["dropped_low_confidence"] + 1
                continue

        kept_words.append(raw_word)

    counters["words_after"] = counters["words_after"] + len(kept_words)

    return " ".join(kept_words)


def clean_ocr_text(
    metadata_path: Path,
    junk_terms_path: Path,
    settings: dict,
    *,
    force: bool = False,
) -> None:

    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} doesn't exist - run OCR first")

    with open(metadata_path, encoding="utf-8") as f:
        metadata_data = json.load(f)

    entries = []
    for e in metadata_data:
        entries.append(VisualMetadata(**e))

    # Stop if every keyframe is already cleaned (cleaned_text None = not done yet)
    if not force:
        all_done = True

        for entry in entries:
            if entry.cleaned_text is None:
                all_done = False

        if all_done:
            print("Nothing to do - every keyframe already has cleaned text.")
            return

    # ---------- Layer 1: which frequent words are interface text? ----------

    junk_words = None

    if not force:
        junk_words = load_saved_junk_words(junk_terms_path, len(entries))

        if junk_words is not None:
            print(f"Reusing the interface-word list from {junk_terms_path.name}: {junk_words}")

    if junk_words is None:
        candidates = find_candidates(entries, settings)
        print(f"{len(candidates)} frequent words are candidates for interface text.")

        if len(candidates) == 0:
            junk_words = []
        else:
            junk_words = ask_gemini_for_interface_words(candidates, entries)

        print(f"Gemini says these are interface words: {junk_words}")

        # Save the evidence so the list can be inspected and checked by eye
        kept_candidates = []
        for word, count in candidates:
            if word not in junk_words:
                kept_candidates.append(word)

        candidate_rows = []
        for word, count in candidates:
            candidate_rows.append({"word": word, "keyframes": count})

        saved = {
            "keyframes_counted": len(entries),
            "model": GEMINI_MODEL,
            "interface_words": junk_words,
            "candidates_kept_as_content": kept_candidates,
            "candidates": candidate_rows,
            "settings": {
                "junk_min_slide_share": settings["junk_min_slide_share"],
                "junk_min_slides": settings["junk_min_slides"],
                "junk_max_candidates": settings["junk_max_candidates"],
            },
        }

        with open(junk_terms_path, "w", encoding="utf-8") as f:
            json.dump(saved, f, indent=2)

    # ---------- Layer 2 + apply both: build cleaned_text for every keyframe ----------

    counters = {
        "words_before": 0,
        "words_after": 0,
        "dropped_symbol": 0,
        "dropped_interface": 0,
        "dropped_low_confidence": 0,
        "keyframes_without_confidences": 0,
    }

    junk_set = set(junk_words)

    for entry in entries:
        entry.cleaned_text = clean_one_entry(entry, junk_set, settings["min_word_confidence"], counters)

    print(f"OCR words: {counters['words_before']} -> {counters['words_after']}")
    print(f"  dropped as symbols (no letters/digits): {counters['dropped_symbol']}")
    print(f"  dropped as interface words:             {counters['dropped_interface']}")
    print(f"  dropped as low confidence (< {settings['min_word_confidence']}):    {counters['dropped_low_confidence']}")

    if counters["keyframes_without_confidences"] > 0:
        print(f"  NOTE: {counters['keyframes_without_confidences']} keyframes have no per-word confidences "
              "(OCR was made before they were saved). Run with --force ocr to add them.")

    # Convert to plain dictionaries before touching the file
    output_data = []

    for entry in entries:
        output_data.append(entry.model_dump())

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)


# Run the program
if __name__ == "__main__":

    from src.lecture_settings import default_settings

    metadata_path = Path(sys.argv[1])
    junk_terms_path = Path(sys.argv[2])

    clean_ocr_text(metadata_path, junk_terms_path, default_settings()["ocr_clean"])
