from faster_whisper import WhisperModel
from src.schemas.knowledge_object import TranscriptSegment
from pathlib import Path
import json

def transcribe(audio_path: Path, transcript_path: Path) -> None:

    results = []
    if transcript_path.exists():
        return

    model = WhisperModel("small", device="cuda", compute_type="float16")


    segments, info = model.transcribe(str(audio_path))

    print(f"Detected Language : {info.language}")
   
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    
    for segment in segments :
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        
        segment_obj = TranscriptSegment(start=segment.start, end=segment.end, text=segment.text)
        results.append(segment_obj)
        
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

        
   
import sys

if __name__ == "__main__":
    audio_path = Path(sys.argv[1])
    transcript_path = Path(sys.argv[2])
    transcribe(audio_path, transcript_path)
