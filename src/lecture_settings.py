# Reads the optional per-lecture settings file and combines it with the defaults from config.py.
#
# Where the file lives:  data/raw/<lecture_name>/settings.json   (optional)
# Anything left out of the file keeps its default value.

import json
from pathlib import Path

from src.config import (
    KEYFRAME_DIFF_THRESHOLD,
    KEYFRAME_CHANGE_AREA_THRESHOLD,
    KEYFRAME_MAX_GAP_SECONDS,
    KEYFRAME_INTERVAL_SECONDS,
)

SETTINGS_FILENAME = "settings.json"


def load_keyframe_settings(lecture_folder: Path) -> dict:
    # Start with the defaults from config.py
    settings = {
        "diff_threshold": KEYFRAME_DIFF_THRESHOLD,
        "change_area_threshold": KEYFRAME_CHANGE_AREA_THRESHOLD,
        "max_gap_seconds": KEYFRAME_MAX_GAP_SECONDS,
        "interval_seconds": KEYFRAME_INTERVAL_SECONDS,
    }

    settings_path = lecture_folder / SETTINGS_FILENAME

    # No file means: use the defaults as they are
    if not settings_path.exists():
        print(f"No {SETTINGS_FILENAME} in {lecture_folder}, using default keyframe settings.")
        return settings

    with open(settings_path, "r", encoding="utf-8") as file:
        file_contents = json.load(file)

    # The file may get other sections later, we only read "keyframes" here
    overrides = file_contents.get("keyframes", {})

    for key in overrides:
        # A misspelled key must be an error. Silently ignoring it would mean
        # you tune a number, nothing changes, and you don't know why.
        if key not in settings:
            allowed_keys = list(settings.keys())
            raise ValueError(f"Unknown keyframe setting '{key}' in {settings_path}. Allowed: {allowed_keys}")

        settings[key] = overrides[key]
        print(f"  override from {SETTINGS_FILENAME}: {key} = {overrides[key]}")

    return settings
