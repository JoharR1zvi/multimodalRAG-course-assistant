# A small web page for asking questions, on top of the same search and answer steps as the
# command line (src/search.py and src/ask.py).
#
# Usage (from the project root):
#   python -m src.api                  then open http://127.0.0.1:8000 in a browser
#   python -m src.api --port 9000
#
# What it serves:
#   GET  /                      the page (src/web/index.html, one file, no build step)
#   GET  /api/lectures          the lectures that can be searched
#   GET  /api/info              the search settings in use (shown in the footer of the page)
#   POST /api/ask               a question in, an answer with sources out (or only the passages)
#   GET  /slides/<lecture>/<f>  a slide picture (keyframe) for a source
#
# Notes:
#   - It listens on 127.0.0.1 only, so it can be reached from this computer and nowhere else.
#   - The database (Qdrant, embedded) can be open in one program at a time, so a question is
#     searched while holding a lock; the call to Gemini happens outside the lock.
#   - The sources shown come from the stored chunks, exactly as on the command line.

import argparse
import re
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src import config
from src.ask import coverage_banner
from src.generation.llm_service import answer_question
from src.retrieval.retriever import format_timestamp, retrieve
from src.search import unique_titles

PROCESSED_DIR = Path("data/processed")
WEB_DIR = Path(__file__).parent / "web"

# How much of the speech of a passage is sent to the page, for the "show the passage" part
EXCERPT_CHARACTERS = 900

# Lecture folder names and picture file names may only contain these characters, so that
# nothing like ../ can ever reach the file system
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
SAFE_PICTURE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.jpg$")

# One search at a time (the embedded database allows one user)
SEARCH_LOCK = threading.Lock()

app = FastAPI(title="Course assistant")


class AskRequest(BaseModel):
    question: str = Field(max_length=1000)
    lecture: str | None = None             # only search this lecture; None = all lectures
    top: int = Field(default=5, ge=1, le=10)
    answer: bool = True                    # False = only show the passages that were found, write no answer


def list_lectures() -> list:
    # The lecture folders that have chunks (so they can be searched)
    if not PROCESSED_DIR.exists():
        return []

    names = []
    for folder in sorted(PROCESSED_DIR.iterdir()):
        if folder.is_dir() and (folder / "chunks.json").exists() and SAFE_NAME.match(folder.name):
            names.append(folder.name)

    return names


def slide_url(chunk) -> str | None:
    # The page address of the picture of the first slide of a passage, or None if it has none
    if len(chunk.image_paths) == 0:
        return None

    # The stored path may use backslashes (written on Windows); only the file name is needed
    file_name = re.split(r"[\\/]", chunk.image_paths[0])[-1]
    if not SAFE_PICTURE_NAME.match(file_name) or not SAFE_NAME.match(chunk.lecture_id):
        return None

    return f"/slides/{chunk.lecture_id}/{file_name}"


def describe_source(number: int, result) -> dict:
    # One passage, as the page needs it. `number` is the citation number in the answer ([1], [2] ...).
    chunk = result.chunk

    excerpt = chunk.text.strip()
    if len(excerpt) > EXCERPT_CHARACTERS:
        excerpt = excerpt[:EXCERPT_CHARACTERS].rsplit(" ", 1)[0] + " ..."

    return {
        "number": number,
        "lecture_id": chunk.lecture_id,
        "start_label": format_timestamp(chunk.start_timestamp),
        "end_label": format_timestamp(chunk.end_timestamp),
        "titles": unique_titles(chunk.slide_titles),
        "image_url": slide_url(chunk),
        "excerpt": excerpt,
        "score": round(result.score, 3),
    }


@app.get("/")
def page():
    return FileResponse(WEB_DIR / "index.html", media_type="text/html")


@app.get("/api/lectures")
def lectures():
    return {"lectures": list_lectures()}


@app.get("/api/info")
def info():
    return {
        "signals": config.RETRIEVAL_SIGNALS,
        "rerank": config.RERANK_ENABLED,
        "embedding_model": config.EMBEDDING_MODEL,
        "answer_model": config.GEMINI_MODEL,
    }


@app.post("/api/ask")
def ask(request: AskRequest):
    question = request.question.strip()
    if question == "":
        raise HTTPException(status_code=400, detail="Type a question first.")

    if request.lecture is not None and request.lecture not in list_lectures():
        raise HTTPException(status_code=400, detail=f"Unknown lecture: {request.lecture}")

    # Search (holding the lock, because the database can be open in one place at a time)
    try:
        with SEARCH_LOCK:
            results = retrieve(question, top_k=request.top, lecture_id=request.lecture)
    except Exception:
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail="The search failed. If another program (for example a command in a terminal) has the database open, close it and try again.",
        )

    if not request.answer:
        return {
            "question": question,
            "mode": "search",
            "sources": [],
            "closest": [describe_source(i + 1, r) for i, r in enumerate(results)],
        }

    try:
        final = answer_question(question, results)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail="The answer could not be written (the Gemini service did not respond). Try again in a moment.")

    sources = []
    for i, source in enumerate(final.sources):
        sources.append(describe_source(final.source_numbers[i], source))

    # When nothing could be answered, show where the search looked, so the student can check
    closest = []
    if not final.answerable:
        for i, result in enumerate(results[:3]):
            closest.append(describe_source(i + 1, result))

    return {
        "question": question,
        "mode": "answer",
        "coverage": final.coverage,
        "answerable": final.answerable,
        "banner": coverage_banner(final.coverage),
        "text": final.text,
        "warnings": final.warnings,
        "sources": sources,
        "closest": closest,
    }


@app.get("/slides/{lecture_id}/{file_name}")
def slide(lecture_id: str, file_name: str):
    if not SAFE_NAME.match(lecture_id) or not SAFE_PICTURE_NAME.match(file_name):
        raise HTTPException(status_code=404)

    path = PROCESSED_DIR / lecture_id / "keyframes" / file_name
    if not path.is_file():
        raise HTTPException(status_code=404)

    return FileResponse(path, media_type="image/jpeg")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the web page for asking questions.")
    parser.add_argument("--port", type=int, default=8000, help="port to listen on (default 8000)")
    args = parser.parse_args()

    print(f"\nOpen http://127.0.0.1:{args.port} in a browser. Stop with Ctrl+C.")
    print("The first question is slower: it loads the search model.\n")

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
