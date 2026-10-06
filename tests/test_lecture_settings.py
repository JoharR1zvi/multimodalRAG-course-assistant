# Tests for load_lecture_settings(): defaults, the optional per-lecture settings.json,
# and the rule that a typo is an error instead of being silently ignored.

import json

import pytest

from src.config import KEYFRAME_CHANGE_AREA_THRESHOLD, OCR_MIN_WORD_CONFIDENCE
from src.lecture_settings import load_lecture_settings


def write_settings_file(folder, contents):
    with open(folder / "settings.json", "w", encoding="utf-8") as f:
        json.dump(contents, f)


def test_without_a_settings_file_everything_is_the_default(tmp_path):
    settings = load_lecture_settings(tmp_path)

    assert settings["keyframes"]["change_area_threshold"] == KEYFRAME_CHANGE_AREA_THRESHOLD
    assert settings["ocr_clean"]["min_word_confidence"] == OCR_MIN_WORD_CONFIDENCE


def test_a_value_in_the_file_replaces_only_that_value(tmp_path):
    write_settings_file(tmp_path, {"keyframes": {"change_area_threshold": 0.15}})

    settings = load_lecture_settings(tmp_path)

    assert settings["keyframes"]["change_area_threshold"] == 0.15
    # Everything the file does not mention keeps its default
    assert settings["keyframes"]["max_gap_seconds"] == load_lecture_settings(tmp_path / "nowhere")["keyframes"]["max_gap_seconds"]
    assert settings["ocr_clean"]["min_word_confidence"] == OCR_MIN_WORD_CONFIDENCE


def test_the_ocr_clean_section_can_be_overridden_too(tmp_path):
    write_settings_file(tmp_path, {"ocr_clean": {"min_word_confidence": 0.1}})

    settings = load_lecture_settings(tmp_path)

    assert settings["ocr_clean"]["min_word_confidence"] == 0.1


def test_a_misspelled_setting_name_is_an_error(tmp_path):
    write_settings_file(tmp_path, {"keyframes": {"change_area_treshold": 0.15}})   # "treshold" typo

    with pytest.raises(ValueError, match="Unknown setting 'change_area_treshold'"):
        load_lecture_settings(tmp_path)


def test_a_misspelled_section_name_is_an_error(tmp_path):
    write_settings_file(tmp_path, {"keyframe": {"change_area_threshold": 0.15}})   # missing the s

    with pytest.raises(ValueError, match="Unknown section 'keyframe'"):
        load_lecture_settings(tmp_path)


def test_loading_settings_twice_does_not_leak_overrides_between_lectures(tmp_path):
    # The defaults dictionary must be built fresh every time. If it were shared, one
    # lecture's override would silently change the next lecture's settings.
    lecture_with_file = tmp_path / "a"
    lecture_without_file = tmp_path / "b"
    lecture_with_file.mkdir()
    lecture_without_file.mkdir()
    write_settings_file(lecture_with_file, {"keyframes": {"change_area_threshold": 0.9}})

    load_lecture_settings(lecture_with_file)
    second = load_lecture_settings(lecture_without_file)

    assert second["keyframes"]["change_area_threshold"] == KEYFRAME_CHANGE_AREA_THRESHOLD
