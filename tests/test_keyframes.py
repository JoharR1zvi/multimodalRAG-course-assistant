# Tests for KeyframeExtractor.compute_difference:
# "what fraction of the screen changed between two frames?"

import numpy as np
import pytest

from src.processing.keyframes import KeyframeExtractor


def make_frame(brightness):
    # A fake video frame: 100 x 100 pixels with 3 colour channels (blue, green, red).
    # Every pixel gets the same brightness (0 = black, 255 = white).
    return np.full((100, 100, 3), brightness, dtype=np.uint8)


def test_identical_frames_have_no_difference():
    extractor = KeyframeExtractor()
    frame = make_frame(0)

    result = extractor.compute_difference(frame, frame)

    assert result == 0.0


def test_black_versus_white_is_a_total_change():
    extractor = KeyframeExtractor()

    result = extractor.compute_difference(make_frame(0), make_frame(255))

    assert result == 1.0


def test_half_the_screen_changing_gives_one_half():
    extractor = KeyframeExtractor()

    frame_before = make_frame(0)

    # Same frame, but the top 50 of the 100 rows turn white
    frame_after = make_frame(0)
    frame_after[:50, :, :] = 255

    result = extractor.compute_difference(frame_before, frame_after)

    # pytest.approx is used because computers store decimals with tiny rounding errors
    assert result == pytest.approx(0.5)


def test_tiny_brightness_changes_are_ignored():
    # The default per-pixel threshold is 40. A pixel going from 100 to 120 changes by only 20,
    # so it does not count as "changed". (This is what makes video noise harmless.)
    extractor = KeyframeExtractor()

    result = extractor.compute_difference(make_frame(100), make_frame(120))

    assert result == 0.0


def test_a_big_brightness_change_counts():
    extractor = KeyframeExtractor()

    # 100 -> 160 is a change of 60, which is more than 40
    result = extractor.compute_difference(make_frame(100), make_frame(160))

    assert result == 1.0


def test_a_stricter_pixel_threshold_notices_smaller_changes():
    # With a threshold of 10, the same 100 -> 120 change (20) now counts
    extractor = KeyframeExtractor(diff_threshold=10)

    result = extractor.compute_difference(make_frame(100), make_frame(120))

    assert result == 1.0


def test_the_order_of_the_two_frames_does_not_matter():
    extractor = KeyframeExtractor()

    frame_a = make_frame(0)
    frame_b = make_frame(0)
    frame_b[:30, :, :] = 255

    assert extractor.compute_difference(frame_a, frame_b) == extractor.compute_difference(frame_b, frame_a)
