# multimodalRAG-course-assistant

A learning project: an assistant that understands a whole university course (lecture videos, slides, PDFs) and answers questions using only that material, with a citation to the lecture, the slide and the moment in the video.

I'm building it one stage at a time and writing down why I made each choice. The reasoning and the measured results are in [`docs/`](docs/).

## Status

**Phase 1 is complete: a lecture video becomes structured, timestamped knowledge.** It has been run on three lectures of two different kinds (annotated slides with a webcam overlay, and a screen recording of live coding). An exit check, `python -m src.verify --all`, passes on all three: every piece of speech is matched to the slide on screen, and no speech is lost.

**Phase 2 is planned, not started:** chunking, embeddings, a vector database, retrieval with an evaluation set, and cited answers. The design is in [`docs/phase2-plan.md`](docs/phase2-plan.md).

## What the pipeline does

```
video --> audio --> transcript (speech with timestamps) -----------------.
  |                                                                      v
  '--> keyframes --> OCR --> vision model --> text cleanup -------> align --> knowledge objects
       (one picture   (text)   (figures, code,    (remove interface    (which slide    (one record per slide:
        per slide)               title, slide no.)  and garbage text)    was showing?)   slide + speech + times)
```

1. **Audio:** FFmpeg extracts the sound track.
2. **Transcript:** Faster-Whisper turns speech into text with timestamps.
3. **Keyframes:** OpenCV finds the moments where the screen changes and saves a picture of each.
4. **OCR:** Tesseract reads the text on each picture.
5. **Vision:** a vision model (Gemini, free tier) describes figures and code, and returns the title and slide number, using a fixed reply form.
6. **Clean:** interface text (menus) and low-confidence garbage are removed from the OCR text.
7. **Align:** each piece of speech is matched to the slide that was on screen.
8. **Knowledge objects:** one JSON record per slide, with its text, description, speech and exact start and end times.

Every stage reads files and writes files, so each result can be opened and checked. How each stage works, and why, is in [`docs/architecture.md`](docs/architecture.md).

## Results so far

| | Length | Keyframes | Words in transcript = knowledge objects |
|---|---|---|---|
| Lecture 1: annotated slides, webcam overlay | 88 min | 64 | 11,194 |
| Lecture 2: same style | 92 min | 67 | 11,192 |
| Lecture 3: public code screencast | 51 min | 73 | 10,051 |

A lecture takes about 10 minutes end to end on a laptop GPU. The vision stage is limited by the free API quota (15 requests per minute). Measured details are in [`docs/decisions.md`](docs/decisions.md).

## Setup

You need Python 3.11, [FFmpeg](https://ffmpeg.org/) and [Tesseract](https://github.com/tesseract-ocr/tesseract) installed, and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

```
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in `GEMINI_API_KEY` (and `TESSERACT_CMD` if Tesseract is not on your PATH). Speech recognition is currently set to run on an NVIDIA GPU, which also needs the cuBLAS and cuDNN libraries on the PATH. Running without a GPU needs a small change in `src/processing/speech.py`.

## Running it

Put one video in `data/raw/<lecture_name>/`, then from the project root:

```
python -m src.pipeline lecture_01                  # one lecture
python -m src.pipeline --all                       # every folder in data/raw/
python -m src.pipeline lecture_01 --force vision   # redo a stage
```

Results appear in `data/processed/<lecture_name>/`, and the main one is `knowledge_objects.json`. Finished stages are skipped, so running again is quick.

To check a finished lecture, or look up what was on screen and said at any second:

```
python -m src.verify lecture_01                # is everything complete and consistent?
python -m src.verify lecture_01 --at 1420      # what was shown and said at 23:40?
```

A lecture can have its own settings in an optional `data/raw/<lecture_name>/settings.json` (for example, a code screencast needs different slide-change settings than a slide deck). See [`docs/architecture.md`](docs/architecture.md).

## Tests

```
python -m pytest
```

65 tests run in about 3 seconds, with no video and no API calls.

## Repository layout

```
src/
  pipeline.py          runs all stages for one lecture or all lectures
  config.py            settings and defaults
  lecture_settings.py  per-lecture settings.json
  verify.py            the Phase 1 exit check
  ingestion/           audio extraction
  processing/          speech, keyframes, OCR, vision, cleanup, alignment, knowledge objects
  schemas/             the data shapes passed between stages
tests/                 automated tests
docs/                  architecture notes, decision log, roadmap, Phase 2 plan
```

## Limits to know about

- Only lecture video is processed so far. PDFs and PowerPoint files are planned.
- Whisper runs on an NVIDIA GPU as configured.
- OCR is weak on terminal and code text, so for code the vision model's text is the useful source.
- Course material is copyrighted, so no lecture videos, slides, transcripts or images are in this repository. The `data/` folder is ignored by git.
