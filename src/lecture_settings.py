# Reads the optional per-lecture settings file and combines it with the defaults from config.py.
#
# Where the file lives:  data/raw/<lecture_name>/settings.json   (optional)
# Anything left out of the file keeps its default value.
#
# Example file:
#   {
#       "keyframes": { "change_area_threshold": 0.15, "max_gap_seconds": 90 },
#       "ocr_clean": { "min_word_confidence": 0.5 }
#   }

import json
from pathlib import Path

from src.config import (
    KEYFRAME_DIFF_THRESHOLD,
    KEYFRAME_CHANGE_AREA_THRESHOLD,
    KEYFRAME_MAX_GAP_SECONDS,
    KEYFRAME_INTERVAL_SECONDS,
    OCR_MIN_WORD_CONFIDENCE,
    OCR_JUNK_MIN_SLIDE_SHARE,
    OCR_JUNK_MIN_SLIDES,
    OCR_JUNK_MAX_CANDIDATES,
)

SETTINGS_FILENAME = "settings.json"


def default_settings() -> dict:
    # Every section of the file, with every setting at its default value.
    # This dictionary also defines which section names and keys are allowed.
    return {
        "keyframes": {
            "diff_threshold": KEYFRAME_DIFF_THRESHOLD,
            "change_area_threshold": KEYFRAME_CHANGE_AREA_THRESHOLD,
            "max_gap_seconds": KEYFRAME_MAX_GAP_SECONDS,
            "interval_seconds": KEYFRAME_INTERVAL_SECONDS,
        },
        "ocr_clean": {
            "min_word_confidence": OCR_MIN_WORD_CONFIDENCE,
            "junk_min_slide_share": OCR_JUNK_MIN_SLIDE_SHARE,
            "junk_min_slides": OCR_JUNK_MIN_SLIDES,
            "junk_max_candidates": OCR_JUNK_MAX_CANDIDATES,
        },
    }


def load_lecture_settings(lecture_folder: Path) -> dict:
    # Returns {"keyframes": {...}, "ocr_clean": {...}}: the defaults, with the
    # lecture's settings.json (if it exists) laid on top.
    settings = default_settings()

    settings_path = lecture_folder / SETTINGS_FILENAME

    # No file means: use the defaults as they are
    if not settings_path.exists():
        print(f"No {SETTINGS_FILENAME} in {lecture_folder}, using default settings.")
        return settings

    with open(settings_path, "r", encoding="utf-8") as file:
        file_contents = json.load(file)

    for section_name in file_contents:
        # A misspelled section or key must be an error. Silently ignoring it would mean
        # you tune a number, nothing changes, and you don't know why.
        if section_name not in settings:
            allowed_sections = list(settings.keys())
            raise ValueError(f"Unknown section '{section_name}' in {settings_path}. Allowed: {allowed_sections}")

        overrides = file_contents[section_name]

        for key in overrides:
            if key not in settings[section_name]:
                allowed_keys = list(settings[section_name].keys())
                raise ValueError(f"Unknown setting '{key}' in section '{section_name}' of {settings_path}. Allowed: {allowed_keys}")

            settings[section_name][key] = overrides[key]
            print(f"  override from {SETTINGS_FILENAME}: {section_name}.{key} = {overrides[key]}")

    return settings
