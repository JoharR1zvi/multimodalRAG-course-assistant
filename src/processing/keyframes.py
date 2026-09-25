import cv2
from pathlib import Path

class KeyframeExtractor:
    def __init__(self, diff_threshold: float = 40.0, change_area_threshold: float = 0.05, max_gap_seconds: float = 180.0):
        self.diff_threshold = diff_threshold
        self.change_area_threshold = change_area_threshold
        self.max_gap_seconds = max_gap_seconds


    def sample_frames(self, video_path: Path, interval_seconds: float):
        cap = cv2.VideoCapture(str(video_path))
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

        timestamp = 0.0
        while timestamp < duration:
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ret, frame = cap.read()
            if ret:
                yield timestamp, frame
            timestamp += interval_seconds

        cap.release()

    def compute_difference(self, frame_a, frame_b) -> float:
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_a, gray_b)
        changed_fraction = (diff > self.diff_threshold).mean()
        return changed_fraction

    def extract(self, video_path: Path, output_dir: Path, interval_seconds: float = 5.0) -> None:
        if output_dir.exists() and any(output_dir.iterdir()):
            return
        output_dir.mkdir(parents=True, exist_ok=True)

        last_saved_frame = None
        last_saved_timestamp = None

        for timestamp, frame in self.sample_frames(video_path, interval_seconds):
            score = self.compute_difference(frame, last_saved_frame) if last_saved_frame is not None else None
            time_since_last_save = timestamp - last_saved_timestamp if last_saved_timestamp is not None else None
            
            is_new_slide = (
                last_saved_frame is None
                or score > self.change_area_threshold
                or time_since_last_save >= self.max_gap_seconds
            )

            if score is not None:
                print(f"t={timestamp:.0f}s changed_fraction={score:.4f} {'SAVED' if is_new_slide else ''}")

            if is_new_slide:
                filename = output_dir / f"frame_{timestamp:.2f}.jpg"
                cv2.imwrite(str(filename), frame)
                last_saved_frame = frame
                last_saved_timestamp = timestamp

