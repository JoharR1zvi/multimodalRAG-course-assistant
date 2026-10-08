# The shapes of an answer: what Gemini must return, and what we hand to the user.

from pydantic import BaseModel

from src.schemas.chunk import SearchResult


class GroundedAnswer(BaseModel):
    # The "form" Gemini must fill in (used by llm_service.py). Like the slide analysis form
    # in the vision step: Gemini answers in exactly this shape, so there is no free text to guess about.
    answerable: bool         # False = the excerpts do not contain the answer
    answer: str              # the answer, with [1], [2] ... after each statement (or, if not answerable, one sentence saying so)


class FinalAnswer(BaseModel):
    # What the user sees: the checked answer plus the real sources behind it
    question: str
    answerable: bool
    text: str                        # the answer with citations that point at real excerpts
    sources: list[SearchResult]      # only the excerpts that are really cited, in order of first citation
    source_numbers: list[int]        # the number each source had in the prompt, same order as `sources`
    warnings: list[str]              # problems found while checking the answer (empty = none)
