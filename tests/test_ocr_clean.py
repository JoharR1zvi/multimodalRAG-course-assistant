# Tests for the pure functions in ocr_clean.py (no files and no Gemini calls involved).
# The one function that talks to Gemini, ask_gemini_for_interface_words, is deliberately
# NOT tested here: tests must never spend API quota.

from src.processing.ocr_clean import (
    clean_one_entry,
    find_candidates,
    has_letter_or_digit,
    normalize_word,
)
from src.schemas.knowledge_object import VisualMetadata


def make_entry(words, confidences=None):
    # A keyframe whose raw OCR text is the given words, one per line (like ocr.py writes it)
    if confidences is None:
        confidences = []

    return VisualMetadata(
        timestamp=0.0,
        image_path="frame_0.00.jpg",
        text="\n".join(words),
        word_confidences=confidences,
    )


def new_counters():
    # The same dictionary the real code creates before it cleans a lecture
    return {
        "words_before": 0,
        "words_after": 0,
        "dropped_symbol": 0,
        "dropped_interface": 0,
        "dropped_low_confidence": 0,
        "keyframes_without_confidences": 0,
    }


# ---------- normalize_word and has_letter_or_digit ----------

def test_normalize_lowercases_and_strips_punctuation_from_the_ends():
    assert normalize_word("File.") == "file"
    assert normalize_word("(Options)") == "options"
    assert normalize_word("Layer:") == "layer"


def test_normalize_keeps_punctuation_inside_a_word():
    assert normalize_word("hello.py") == "hello.py"


def test_normalize_turns_a_pure_symbol_into_an_empty_string():
    assert normalize_word("|") == ""


def test_symbols_have_no_letter_or_digit_but_words_and_numbers_do():
    assert has_letter_or_digit("|") is False
    assert has_letter_or_digit(">") is False
    assert has_letter_or_digit("4096") is True
    assert has_letter_or_digit("a") is True


# ---------- find_candidates ----------

SETTINGS = {"junk_min_slide_share": 0.25, "junk_min_slides": 5, "junk_max_candidates": 80}


def make_ten_keyframes():
    # Ten keyframes. Which words appear on how many of them:
    #   file     -> 10     the    -> 8     enough  -> 5 (exactly the minimum)
    #   almost   -> 4 (one too few)       rare    -> 1
    #   "42" (a number) and "of" (too short) -> 10, but they can never be interface words
    # (The test words are plain letters on purpose: only letter-only words are candidates.)
    entries = []

    for number in range(10):
        words = ["file", "42", "of"]

        if number < 8:
            words.append("the")
        if number < 5:
            words.append("enough")
        if number < 4:
            words.append("almost")
        if number == 0:
            words.append("rare")

        entries.append(make_entry(words))

    return entries


def test_frequent_words_become_candidates_with_their_slide_counts():
    candidates = find_candidates(make_ten_keyframes(), SETTINGS)

    # Most frequent first
    assert candidates[0] == ("file", 10)
    assert candidates[1] == ("the", 8)


def test_a_word_needs_enough_slides_to_be_a_candidate():
    candidates = find_candidates(make_ten_keyframes(), SETTINGS)

    candidate_words = []
    for word, count in candidates:
        candidate_words.append(word)

    assert "enough" in candidate_words         # on 5 slides = exactly the minimum
    assert "almost" not in candidate_words     # on 4 slides = one too few
    assert "rare" not in candidate_words


def test_numbers_and_very_short_words_are_never_candidates():
    candidates = find_candidates(make_ten_keyframes(), SETTINGS)

    candidate_words = []
    for word, count in candidates:
        candidate_words.append(word)

    assert "42" not in candidate_words
    assert "of" not in candidate_words


def test_the_candidate_list_is_cut_to_the_maximum_size():
    small_settings = {"junk_min_slide_share": 0.25, "junk_min_slides": 5, "junk_max_candidates": 1}

    candidates = find_candidates(make_ten_keyframes(), small_settings)

    assert candidates == [("file", 10)]


def test_a_word_repeated_on_one_slide_counts_that_slide_only_once():
    # "file" appears three times on the same keyframe, but that is still ONE keyframe
    entries = [make_entry(["file", "file", "file"])]

    candidates = find_candidates(entries, {"junk_min_slide_share": 0.0, "junk_min_slides": 1, "junk_max_candidates": 80})

    assert candidates == [("file", 1)]


# ---------- clean_one_entry ----------

def test_cleaning_removes_interface_words_symbols_and_low_confidence_words():
    entry = make_entry(
        ["File", "Accuracy", "|", "oP", "real"],
        [0.93, 0.96, 0.0, 0.02, 0.90],
    )
    counters = new_counters()

    cleaned = clean_one_entry(entry, {"file"}, 0.3, counters)

    assert cleaned == "Accuracy real"
    assert counters["words_before"] == 5
    assert counters["words_after"] == 2
    assert counters["dropped_interface"] == 1          # File
    assert counters["dropped_symbol"] == 1             # |
    assert counters["dropped_low_confidence"] == 1     # oP


def test_a_word_exactly_at_the_cutoff_is_kept():
    entry = make_entry(["borderline"], [0.3])

    cleaned = clean_one_entry(entry, set(), 0.3, new_counters())

    assert cleaned == "borderline"


def test_interface_words_are_matched_ignoring_case_and_punctuation():
    entry = make_entry(["FILE", "Edit:", "(View)", "content"], [0.9, 0.9, 0.9, 0.9])

    cleaned = clean_one_entry(entry, {"file", "edit", "view"}, 0.3, new_counters())

    assert cleaned == "content"


def test_words_with_no_confidence_number_are_kept():
    # Tesseract sometimes gives no confidence; ocr.py stores that as -1
    entry = make_entry(["mystery"], [-1.0])

    cleaned = clean_one_entry(entry, set(), 0.3, new_counters())

    assert cleaned == "mystery"


def test_a_lower_cutoff_keeps_more_words():
    # This is why the code lecture uses 0.1: real code often scores low
    entry = make_entry(["python", "hello"], [0.2, 0.25])

    strict = clean_one_entry(entry, set(), 0.3, new_counters())
    gentle = clean_one_entry(entry, set(), 0.1, new_counters())

    assert strict == ""
    assert gentle == "python hello"


def test_without_per_word_confidences_only_the_other_rules_apply():
    # An old visual_metadata.json has no word_confidences. We must not crash or drop everything.
    entry = make_entry(["File", "Accuracy", "|"], [])
    counters = new_counters()

    cleaned = clean_one_entry(entry, {"file"}, 0.3, counters)

    assert cleaned == "Accuracy"
    assert counters["keyframes_without_confidences"] == 1


def test_a_keyframe_with_no_text_cleans_to_an_empty_string():
    entry = make_entry([])

    cleaned = clean_one_entry(entry, {"file"}, 0.3, new_counters())

    assert cleaned == ""
