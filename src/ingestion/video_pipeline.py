# Converts a raw lecture video into a clean, Whisper-ready audio file.
# First step in "unlocking" what's inside the video.

import subprocess
from pathlib import Path


def extract_audio(video_path: Path, audio_path: Path, *, force: bool = False) -> None:

    # Stop if the audio already exists
    if audio_path.exists():
        if not force:
            return

    # Make sure the output folder exists
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the FFmpeg command as a list, one piece per item
    command = [
        "ffmpeg",
        "-y",                    # overwrite the output file if it exists
        "-i", str(video_path),   # input file
        "-vn",                   # drop the video, keep only audio
        "-acodec", "pcm_s16le",  # plain uncompressed WAV audio
        "-ar", "16000",          # 16 kHz sample rate (what Whisper expects)
        "-ac", "1",              # 1 channel (mono)
        str(audio_path),         # output file
    ]

    # Run FFmpeg. check=True makes Python raise an error if FFmpeg fails
    subprocess.run(command, check=True)
