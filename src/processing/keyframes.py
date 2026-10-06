# Finds the moments in the video where the slide changes and saves a picture of each.

import cv2
from pathlib import Path
from src.config import (
    KEYFRAME_DIFF_THRESHOLD,
    KEYFRAME_CHANGE_AREA_THRESHOLD,
    KEYFRAME_MAX_GAP_SECONDS,
    KEYFRAME_INTERVAL_SECONDS,
)



class KeyframeExtractor:

    def __init__(
        self,
        diff_threshold: float = KEYFRAME_DIFF_THRESHOLD,
        change_area_threshold: float = KEYFRAME_CHANGE_AREA_THRESHOLD,
        max_gap_seconds: float = KEYFRAME_MAX_GAP_SECONDS,

    ):
        # How much ONE pixel must change (0-255) to count as "changed"
        self.diff_threshold = diff_threshold

        # What fraction of the WHOLE frame must have changed to call it a new slide
        self.change_area_threshold = change_area_threshold

        # Safety net: force a save if this many seconds pass with no detected change
        self.max_gap_seconds = max_gap_seconds

    def sample_frames(self, video_path: Path, interval_seconds: float):
        # This is a "generator": it uses yield instead of return, so it hands back
        # one (timestamp, frame) pair at a time instead of loading the whole video into memory.
        cap = cv2.VideoCapture(str(video_path))

        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps

        timestamp = 0.0

        while timestamp < duration:
            # Jump straight to this time (much faster than decoding every frame)
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

            success, frame = cap.read()

            if success:
                yield timestamp, frame

            timestamp = timestamp + interval_seconds

        cap.release()

    def compute_difference(self, frame_a, frame_b) -> float:
        # Convert both frames to grayscale (one number per pixel instead of three)
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

        # Per-pixel absolute difference between the two frames
        diff = cv2.absdiff(gray_a, gray_b)

        # True where the pixel changed a lot, False elsewhere.
        # .mean() of True/False values = the fraction of pixels that changed
        changed_fraction = (diff > self.diff_threshold).mean()

        return changed_fraction

    def extract(
        self,
        video_path: Path,
        output_dir: Path,
        *,
        force: bool = False,
        interval_seconds: float = KEYFRAME_INTERVAL_SECONDS,
    ) -> None:

        # Stop if the folder already has keyframes in it
        if output_dir.exists():
            if any(output_dir.iterdir()):
                if not force:
                    return

        output_dir.mkdir(parents=True, exist_ok=True)

        # The last frame we saved, and when. Start empty
        last_saved_frame = None
        last_saved_timestamp = None

        for timestamp, frame in self.sample_frames(video_path, interval_seconds):

            if last_saved_frame is None:
                # Very first frame: always save it
                score = None
                is_new_slide = True
            else:
                # Compare this frame to the last SAVED frame (not the previous sampled one)
                score = self.compute_difference(frame, last_saved_frame)
                time_since_last_save = timestamp - last_saved_timestamp

                # Save if enough of the screen changed, OR too much time has passed
                is_new_slide = False

                if score > self.change_area_threshold:
                    is_new_slide = True

                if time_since_last_save >= self.max_gap_seconds:
                    is_new_slide = True

            # Debug print, useful when tuning the thresholds
            if score is not None:
                if is_new_slide:
                    label = "SAVED"
                else:
                    label = ""
                print(f"t={timestamp:.0f}s changed_fraction={score:.4f} {label}")

            if is_new_slide:
                filename = output_dir / f"frame_{timestamp:.2f}.jpg"
                cv2.imwrite(str(filename), frame)

                last_saved_frame = frame
                last_saved_timestamp = timestamp
