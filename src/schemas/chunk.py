# The shape of one retrieval chunk: the piece of a lecture that gets embedded and searched.
# Written by chunking.py into chunks.json, read by the embedding, search and answer steps.

from pydantic import BaseModel


class Chunk(BaseModel):
    # Who it is. The id is built from the lecture and the start second, so building the
    # chunks again gives the same ids (re-indexing then overwrites instead of duplicating).
    chunk_id: str
    course_id: str
    lecture_id: str

    # Where it is. Times come from the exact start/end of the speech pieces inside the chunk.
    start_timestamp: float
    end_timestamp: float
    slide_timestamps: list[float]        # start time of every slide this chunk covers, in time order
    image_paths: list[str]               # the keyframe image of each of those slides
    slide_titles: list[str]              # titles of those slides ("" ones left out)

    # What it says: the speech, and all three versions of the slide text (joined over the
    # slides above) so retrieval experiments can swap them without rebuilding anything.
    text: str                            # everything said in this chunk
    slide_text: str                      # raw OCR text
    cleaned_text: str                    # OCR text after the cleanup stage
    clean_text: str                      # the vision model's reading of the slide text
    slide_description: str                # the vision model's description of diagrams / code

    # The exact string that gets embedded (slide text + description + speech)
    embed_text: str

    # Neighbours, so a hit can later be widened with the chunk before or after it (None at the ends)
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
