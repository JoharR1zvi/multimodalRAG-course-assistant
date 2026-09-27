import pytesseract
from PIL import Image
from src.config import TESSERACT_CMD
from src.schemas.knowledge_object import VisualMetadata
from pathlib import Path
import json

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def _parse_timestamp(image_path: Path) -> float:
    return float(image_path.stem.removeprefix("frame_"))

def extract_text(keyframes_dir: Path, output_path: Path, *, force: bool = False) -> None:
    if output_path.exists() and not force:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(keyframes_dir.glob("*.jpg"), key=_parse_timestamp)

    results = []
    for image_path in image_paths:
        timestamp = _parse_timestamp(image_path)
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        lines = [text for text in data["text"] if text.strip()]
        confidences = [float(conf) for conf, text in zip(data["conf"], data["text"]) if text.strip() and float(conf) >= 0]

        metadata = VisualMetadata(
            timestamp=timestamp,
            image_path=str(image_path),
            text="\n".join(lines),
            confidence=(sum(confidences) / len(confidences) / 100.0) if confidences else None,
        )
        results.append(metadata)
        print(f"[{timestamp:.2f}s] {image_path.name}: {len(lines)} lines, confidence={metadata.confidence}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in results], f, indent=2)

import sys
if __name__ == "__main__":
    keyframes_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    extract_text(keyframes_dir, output_path)
