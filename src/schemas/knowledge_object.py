from pydantic import BaseModel

class TranscriptSegment (BaseModel):
    start:float
    end:float
    text:str
    