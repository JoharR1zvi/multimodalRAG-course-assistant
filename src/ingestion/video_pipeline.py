#converts a raw lecture video into a clean, Whisper-ready audio file. The first step in "unlocking" what's inside the video.


import subprocess 
from pathlib import Path

def extract_audio(video_path : Path , audio_path : Path,*,force:bool=False) -> None:
    if audio_path.exists() and not force:
        return
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
         "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),       
    ], check=True)