# The shapes of an answer: what Gemini must return, and what we hand to the user.

from typing import Literal

from pydantic import BaseModel

from src.schemas.chunk import SearchResult


class GroundedAnswer(BaseModel):
    # The "form" Gemini must fill in (used by llm_service.py). Like the slide analysis form
    # in the vision step: Gemini answers in exactly this shape, so there is no free text to guess about.
    # The ORDER matters: the answer is written first, and then the model says how well the
    # excerpts covered the question, so the label agrees with what was just written.
    answer: str                                      # the answer, with [1], [2] ... after each statement (or, if nothing is covered, one sentence saying so)
    coverage: Literal["full", "partial", "none"]     # full = the excerpts answer all of it, partial = only part of it, none = nothing relevant


class FinalAnswer(BaseModel):
    # What the user sees: the checked answer plus the real sources behind it
    question: str
    answerable: bool                 # True when the excerpts cover at least part of the question (coverage is not "none")
    coverage: str                    # "full", "partial" or "none"
    text: str                        # the answer with citations that point at real excerpts
    sources: list[SearchResult]      # only the excerpts that are really cited, in order of first citation
    source_numbers: list[int]        # the number each source had in the prompt, same order as `sources`
    warnings: list[str]              # problems found while checking the answer (empty = none)
