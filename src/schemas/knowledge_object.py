from pydantic import BaseModel

class TranscriptSegment (BaseModel):
    start:float
    end:float
    text:str
    
class VisualMetadata(BaseModel):
    timestamp:float 
    image_path: str
    text: str 
    confidence: float | None= None
    description: str | None = None

    