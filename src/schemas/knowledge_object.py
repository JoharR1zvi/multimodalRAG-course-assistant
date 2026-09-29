# The shapes (schemas) of the data that flows between pipeline stages.
# Pydantic checks every field's type when an object is created,
# so bad data fails immediately instead of breaking a later stage.

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    # One piece of speech, from speech.py
    start: float
    end: float
    text: str


class VisualMetadata(BaseModel):
    # One keyframe (slide), from ocr.py and vision.py
    timestamp: float
    image_path: str
    text: str                            # text read by OCR
    confidence: float | None = None      # OCR confidence, 0 to 1
    description: str | None = None       # Gemini's diagram description ("" = no diagram, None = not done yet)


class AlignedSegment(BaseModel):
    # One piece of speech plus which slide was showing, from alignment.py
    start: float
    end: float
    text: str
    keyframe_timestamp: float | None     # None = spoken before the first slide


class KnowledgeObject(BaseModel):
    # One slide plus everything said while it was showing, from knowledge.py
    lecture_id: str
    start_timestamp: float
    end_timestamp: float
    transcript: str
    slide_text: str
    slide_description: str
    image_path: str
