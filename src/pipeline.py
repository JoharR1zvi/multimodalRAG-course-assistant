# The "conductor": runs one lecture (or all lectures) through every stage, in order.
# This file has NO processing logic of its own. It only calls the stages we already built
# with the right file paths, and times each one.
#
# Usage (from the project root):
#   python -m src.pipeline lecture_01                       # one lecture
#   python -m src.pipeline --all                            # every folder inside data/raw/
#   python -m src.pipeline lecture_01 --force ocr           # redo one stage
#   python -m src.pipeline lecture_01 --force ocr --force vision   # redo several stages
#
# Folder convention:
#   data/raw/<lecture_name>/<video file>        <- input, never modified
#   data/processed/<lecture_name>/...           <- every output lands here

import argparse
import time
from pathlib import Path

from src.ingestion.video_pipeline import extract_audio
from src.processing.speech import transcribe
from src.processing.keyframes import KeyframeExtractor
from src.processing.ocr import extract_text
from src.processing.vision import describe_images
from src.processing.alignment import align
from src.processing.knowledge import build_knowledge_objects

from src.lecture_settings import load_keyframe_settings
from src.config import DELETE_AUDIO_AFTER_TRANSCRIPT


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# The stage names you are allowed to pass to --force, in pipeline order
STAGE_NAMES = ["audio", "transcript", "keyframes", "ocr", "vision", "align", "knowledge"]

# File types we accept as a lecture video
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi", ".webm"]


def find_video(lecture_name: str) -> Path:
    # Look inside data/raw/<lecture_name>/ for exactly one video file

    lecture_folder = RAW_DIR / lecture_name

    if not lecture_folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {lecture_folder}")

    videos = []

    for file in lecture_folder.iterdir():
        if file.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(file)

    # Refuse to guess if there is nothing, or more than one candidate
    if len(videos) == 0:
        raise FileNotFoundError(f"No video file found in {lecture_folder}")

    if len(videos) > 1:
        names = []
        for video in videos:
            names.append(video.name)
        raise ValueError(f"More than one video in {lecture_folder}: {names}. Keep exactly one.")

    return videos[0]


def run_stage(stage_name: str, timings: list, stage_function, *args, **kwargs) -> None:
    # Runs ONE stage and records how long it took.
    #   stage_function : the function to call, e.g. extract_audio
    #   *args, **kwargs: whatever should be passed on to it (paths, force=...)
    # Every stage call looks the same, so this helper saves us writing the timer code 7 times.

    print(f"\n=== {stage_name} ===")

    start_time = time.time()

    stage_function(*args, **kwargs)

    elapsed = time.time() - start_time

    print(f"--- {stage_name} finished in {elapsed:.1f}s")
    timings.append((stage_name, elapsed))


def process_lecture(lecture_name: str, force_stages: list) -> list:
    # Runs all 7 stages for one lecture. Returns the list of (stage_name, seconds) timings.

    video_path = find_video(lecture_name)
    out_dir = PROCESSED_DIR / lecture_name

    print(f"\n##### Processing {lecture_name} (video: {video_path.name}) #####")
    # Load the per-lecture settings FIRST, so a typo in settings.json fails
    # immediately instead of after the slow transcription stage
    keyframe_settings = load_keyframe_settings(RAW_DIR / lecture_name)


    # Warn about a costly combination: forcing OCR recreates visual_metadata.json,
    # which wipes every Gemini description, so vision then has to call Gemini for all keyframes again
    if "ocr" in force_stages and "vision" not in force_stages:
        print("NOTE: --force ocr wipes the Gemini descriptions, so vision will re-describe every keyframe.")

    # All the file paths, in one place
    audio_path = out_dir / "audio.wav"
    transcript_path = out_dir / "transcript.json"
    keyframes_dir = out_dir / "keyframes"
    visual_metadata_path = out_dir / "visual_metadata.json"
    alignment_path = out_dir / "alignment.json"
    knowledge_path = out_dir / "knowledge_objects.json"

    timings = []

    # The audio file only exists to make the transcript. So when transcript.json is already
    # there, we do not need the audio at all, unless someone forces audio or transcript.
    transcript_exists = transcript_path.exists()

    audio_needed = False

    if not transcript_exists:
        audio_needed = True

    if "transcript" in force_stages:
        audio_needed = True

    if "audio" in force_stages:
        audio_needed = True

    # Stage 1: video -> audio
    if audio_needed:
        run_stage(
            "audio", timings, extract_audio,
            video_path, audio_path,
            force=("audio" in force_stages),
        )
    else:
        print("\n=== audio ===")
        print("--- skipped: transcript.json already exists, so the audio file is not needed")

    # Stage 2: audio -> transcript (the slowest stage, uses the GPU)
    run_stage(
        "transcript", timings, transcribe,
        audio_path, transcript_path,
        force=("transcript" in force_stages),
    )

    # The audio file is big and can be recreated from the video, so remove it
    # as soon as the transcript exists (can be switched off in config / .env)
    if DELETE_AUDIO_AFTER_TRANSCRIPT:
        if transcript_path.exists() and audio_path.exists():
            audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
            audio_path.unlink()
            print(f"Deleted {audio_path.name} ({audio_size_mb:.0f} MB). The transcript is done; the audio can be recreated from the video.")

    # Stage 3: video -> keyframes. The settings (defaults from config.py + optional
    # settings.json) were already loaded at the top of this function.
    extractor = KeyframeExtractor(
        diff_threshold=keyframe_settings["diff_threshold"],
        change_area_threshold=keyframe_settings["change_area_threshold"],
        max_gap_seconds=keyframe_settings["max_gap_seconds"],
    )
    run_stage(
        "keyframes", timings, extractor.extract,
        video_path, keyframes_dir,
        force=("keyframes" in force_stages),
        interval_seconds=keyframe_settings["interval_seconds"],
    )


    # Stage 4: keyframes -> OCR text
    run_stage(
        "ocr", timings, extract_text,
        keyframes_dir, visual_metadata_path,
        force=("ocr" in force_stages),
    )

    # Stage 5: add Gemini descriptions to visual_metadata.json (edits the file in place).
    # It skips keyframes that already have a description, so it is cheap when everything is done
    run_stage(
        "vision", timings, describe_images,
        visual_metadata_path,
        force=("vision" in force_stages),
    )

    # Stage 6: match each speech segment to the slide showing at that moment
    run_stage(
        "align", timings, align,
        transcript_path, visual_metadata_path, alignment_path,
        force=("align" in force_stages),
    )

    # Stage 7: merge everything into one knowledge object per slide
    run_stage(
        "knowledge", timings, build_knowledge_objects,
        alignment_path, visual_metadata_path, knowledge_path,
        force=("knowledge" in force_stages),
    )

    return timings


def print_timings(lecture_name: str, timings: list) -> None:
    # A small summary table at the end of a lecture

    print(f"\n----- Timing summary for {lecture_name} -----")

    total = 0.0

    for stage_name, seconds in timings:
        print(f"  {stage_name:<12} {seconds:8.1f}s")
        total = total + seconds

    print(f"  {'TOTAL':<12} {total:8.1f}s")


def main() -> None:

    # argparse reads what you typed after the command and checks it is valid
    parser = argparse.ArgumentParser(description="Process lecture videos into knowledge objects.")

    # "?" means the lecture name is optional (you can use --all instead)
    parser.add_argument("lecture", nargs="?", help="folder name inside data/raw/, e.g. lecture_01")
    parser.add_argument("--all", action="store_true", help="process every lecture folder in data/raw/")
    parser.add_argument(
        "--force",
        action="append",
        choices=STAGE_NAMES,
        default=[],
        help="redo this stage even if its output exists (repeat the flag for several stages)",
    )

    args = parser.parse_args()

    # Exactly one of: a lecture name, or --all
    if args.all and args.lecture:
        parser.error("give a lecture name OR --all, not both")

    if not args.all and not args.lecture:
        parser.error("give a lecture name, or use --all")

    # One lecture: any error stops the program and shows the error (easier to debug)
    if not args.all:
        timings = process_lecture(args.lecture, args.force)
        print_timings(args.lecture, timings)
        return

    # --all: one broken lecture must not stop the rest, so catch errors per lecture
    lecture_names = []

    for folder in sorted(RAW_DIR.iterdir()):
        if folder.is_dir():
            lecture_names.append(folder.name)

    failures = []

    for lecture_name in lecture_names:
        try:
            timings = process_lecture(lecture_name, args.force)
            print_timings(lecture_name, timings)
        except Exception as error:
            print(f"\n!!!!! {lecture_name} FAILED: {error}")
            failures.append(lecture_name)

    print(f"\nDone. {len(lecture_names) - len(failures)} of {len(lecture_names)} lectures succeeded.")

    if len(failures) > 0:
        print(f"Failed: {failures}")


if __name__ == "__main__":
    main()
