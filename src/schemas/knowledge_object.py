# The shapes (schemas) of the data that flows between pipeline stages.
# Pydantic checks every field's type when an object is created,
# so bad data fails immediately instead of breaking a later stage.

from typing import Literal

from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    # One piece of speech, from speech.py
    start: float
    end: float
    text: str


class SlideAnalysis(BaseModel):
    # The "form" Gemini must fill in for every keyframe (used by vision.py).
    # Gemini is forced to answer in exactly this shape, so there is no free text to guess about.
    #
    # ORDER MATTERS: Gemini writes the boxes top to bottom. description comes BEFORE
    # content_type on purpose, so the model has already looked closely at the picture and
    # written what it sees by the time it has to pick the type.
    title: str                  # the slide's title, "" if it has none
    description: str            # visuals/code only: what it shows (code: the code + one-line summary). Text-only slide = ""
    content_type: Literal[
        "text_slide",     # ONLY text/bullets/formulas, no figure at all
        "title_slide",    # a section or lecture title
        "diagram",        # any figure, drawing, flowchart, architecture picture (even if text is also on the slide)
        "chart",          # a plot with axes, or a picture of a table of results
        "code",           # source code, terminal, notebook
        "equation",       # the slide is mainly mathematical formulas
        "other",          # anything that does not fit above
    ]
    slide_number: int | None    # page/slide number if visible on screen, else None
    clean_text: str             # all the teaching text on screen, without menus/toolbars


class JunkTermsAnswer(BaseModel):
    # The form Gemini fills in when asked which frequent OCR words are software interface text
    # (used by ocr_clean.py)
    interface_words: list[str]


class VisualMetadata(BaseModel):
    # One keyframe (slide), from ocr.py and vision.py
    timestamp: float
    image_path: str
    text: str                            # text read by OCR
    confidence: float | None = None      # OCR confidence, 0 to 1
    description: str | None = None       # Gemini's description ("" = nothing to describe, None = not done yet)

    # The fields below are filled by vision.py. They have defaults so that an
    # older visual_metadata.json (written before these existed) still loads.
    content_type: str | None = None      # one of the SlideAnalysis types, None = not done yet
    title: str = ""                      # slide title according to Gemini
    slide_number: int | None = None      # slide/page number if Gemini could see one
    clean_text: str = ""                 # Gemini's version of the slide text (kept next to the OCR text to compare)

    # The fields below are filled by ocr.py and ocr_clean.py (OCR cleanup).
    # `text` above is never changed, so the raw OCR stays available for comparison.
    word_confidences: list[float] = []   # one number per word in `text` (same order), 0 to 1, -1 = none given
    cleaned_text: str | None = None      # OCR text after removing interface words and low-confidence junk (None = not done yet)


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

    # From vision.py. Defaults let an older knowledge_objects.json still load.
    content_type: str = ""               # text_slide / diagram / code / ... ("" = unknown)
    title: str = ""                      # slide title ("" = none)
    slide_number: int | None = None      # slide/page number if visible on screen
