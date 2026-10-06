# Roadmap

Last updated: 2026-10-06.

## Where things stand

Phase 1 (one lecture video into structured, timestamped knowledge) works end to end on the first test lecture. One command runs all seven stages:

```
python -m src.pipeline lecture_01
```

Verified on that lecture: 64 knowledge objects, each slide's end time equals the next slide's start, and no speech is lost between the transcript (11,194 words) and the knowledge objects (11,194 words).

## Before Phase 2: checklist

Each step is small and checked against the first lecture before moving on.

1. **Keyframe thresholds into config, with a per-lecture override.** *Done (2026-10-06).* Defaults live in `config.py`; an optional `settings.json` next to a lecture's video overrides them. Re-extracting into a scratch folder gave the same 64 filenames. See decision 13.
2. **Layout-agnostic vision prompt with structured output** (`content_type`, `title`, `description`, `slide_number`, clean slide text). *Done (2026-10-06), applied to the first lecture.* No earlier diagram lost, text-only slides no longer get descriptions, nothing failed. Also added a request-rate limiter and worker threads. See decision 14.
3. **Delete `audio.wav` after the transcript exists**, as a config option, with the matching skip logic in the pipeline. *Done (2026-10-06).* See decision 16.
4. **Process a second lecture that looks different from the first**: a code screencast, then a clean slide-deck recording. *Partly done (2026-10-06).* Lecture 2 (same style) and lecture 3 (a public code screencast) both ran end to end, and lecture 3 needed its own settings file. See decision 17. Not yet tried: a clean slide-deck recording without annotations or a webcam.
5. **Real timings**, then speedups where the numbers justify them. *Timings done (2026-10-06), see decision 17.* Vision (quota-limited) and speech recognition are about 85% of the time. Remaining speed ideas are in "Speed work".
6. **OCR junk cleanup layers.** *Done (2026-10-06).* A new `clean` stage writes `cleaned_text` next to the raw OCR text, so raw OCR, cleaned OCR and the vision model's text can be compared fairly at retrieval time (decisions 11 and 15). Also fixed the label for camera shots in the code lecture.
7. **First automated tests**: `align`, the grouping logic in `knowledge.py`, and `compute_difference`, on small hand-made data. *Done (2026-10-06).* 43 tests run in about 3 seconds with `python -m pytest`, with no video and no API calls. They also cover the OCR cleaning functions and the settings loader. To check that the tests can fail, I broke three small things on purpose (an off-by-one at the moment a slide appears, a cutoff comparison, a missing `.strip()`), and each break was caught by the matching test. Not covered, because they need a video, Tesseract or the API: speech, keyframe extraction end to end, OCR, the vision step and the pipeline runner.
8. **Documentation pass** and publishing of the public docs.
9. **Phase 1 exit check** on both lectures. Then Phase 2 opens.

## Target video types

Slides with a presenter, and code or screen demos. Whiteboard and blackboard recordings are out of scope for now.

## Phase 2 (planned)

Chunking, embeddings (local model and API, compared), vector store, retrieval with evaluation, optional reranking, cited answer generation. Details in `phase2-plan.md`.

## Later

Backend API and a user interface. Support for PDFs and PowerPoint files as course material (currently only lecture video is processed).

## Phase 2 decision to remember: which slide text gets embedded

Compare three versions on the evaluation set, changing only the embedded text: raw OCR, cleaned OCR, and the vision model's clean text. Details in decision 15.

## Automatic keyframe settings (after the code-screencast lecture)

A hand-written `settings.json` is for developers. Plan: pick a preset from a few sampled frames (slides or code demo), and/or measure the video's normal frame-to-frame noise and set the threshold from it, and save the settings each run used. The file stays as the final override. See decision 13.

## Speed work (after real timings exist)

- Gemini calls: done as a rate limiter plus small thread pool. The free tier allows 15 requests per minute, so a lecture of 64 keyframes takes about 5 minutes and cannot go much faster without a higher quota (a `.env` setting) or several frames per request.
- OCR images across CPU cores: skipping this, OCR takes under a minute per lecture (decision 17).
- Run the GPU stage (speech) at the same time as the CPU and network stages for the same lecture, and pipeline across lectures so the GPU never waits.
- Speech model tuning if it dominates: batched inference, voice activity filtering, compute type. Any change is checked against the current transcript first.
- Last, because it changes output: keyframe sampling interval and frame downscaling, which need re-tuning.
