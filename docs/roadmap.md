# Roadmap

Last updated: 2026-10-08.

## Where things stand

**Phase 2 has a working prototype (2026-10-08).** Chunking, embeddings (bge-m3 on the GPU), an embedded Qdrant database, search, and answers with citations all work end to end on three lectures: 96 chunks are stored, `python -m src.search` finds the closest chunks, and `python -m src.ask` answers from them with sources that come from stored data. 144 tests pass. What is missing is measurement: an evaluation set is built (11 questions with known answer locations and 3 questions the lectures do not cover, kept private), but the command that runs it is not written yet, so search quality is not known. The overview is in the README, with diagrams, and the status of every Phase 2 step is in `phase2-plan.md`.

**Phase 1 is complete (2026-10-06).** One lecture video becomes structured, timestamped knowledge, and it has been run on three lectures of two kinds (annotated slides with a webcam overlay, and a screen recording of live coding). One command runs all eight stages:

```
python -m src.pipeline lecture_01
```

A second command, `python -m src.verify --all`, is the Phase 1 exit check. It passes on all three lectures: every file is complete and agrees with the others, each slide's end time equals the next slide's start, no speech is lost between the transcript and the knowledge objects, and every piece of speech lies on a slide. Details in decision 18.

## Before Phase 2: checklist

Each step was small and checked against the first lecture before moving on. All nine are done.

1. **Keyframe thresholds into config, with a per-lecture override.** *Done (2026-10-06).* Defaults live in `config.py`; an optional `settings.json` next to a lecture's video overrides them. Re-extracting into a scratch folder gave the same 64 filenames. See decision 13.
2. **Layout-agnostic vision prompt with structured output** (`content_type`, `title`, `description`, `slide_number`, clean slide text). *Done (2026-10-06), applied to the first lecture.* No earlier diagram lost, text-only slides no longer get descriptions, nothing failed. Also added a request-rate limiter and worker threads. See decision 14.
3. **Delete `audio.wav` after the transcript exists**, as a config option, with the matching skip logic in the pipeline. *Done (2026-10-06).* See decision 16.
4. **Process a second lecture that looks different from the first**: a code screencast, then a clean slide-deck recording. *Partly done (2026-10-06).* Lecture 2 (same style) and lecture 3 (a public code screencast) both ran end to end, and lecture 3 needed its own settings file. See decision 17. Not yet tried: a clean slide-deck recording without annotations or a webcam.
5. **Real timings**, then speedups where the numbers justify them. *Timings done (2026-10-06), see decision 17.* Vision (quota-limited) and speech recognition are about 85% of the time. Remaining speed ideas are in "Speed work".
6. **OCR junk cleanup layers.** *Done (2026-10-06).* A new `clean` stage writes `cleaned_text` next to the raw OCR text, so raw OCR, cleaned OCR and the vision model's text can be compared fairly at retrieval time (decisions 11 and 15). Also fixed the label for camera shots in the code lecture.
7. **First automated tests**: `align`, the grouping logic in `knowledge.py`, and `compute_difference`, on small hand-made data. *Done (2026-10-06).* 65 tests (43 at first, 22 more for the exit check in step 9) run in about 3 seconds with `python -m pytest`, with no video and no API calls. They also cover the OCR cleaning functions and the settings loader. To check that the tests can fail, I broke three small things on purpose (an off-by-one at the moment a slide appears, a cutoff comparison, a missing `.strip()`), and each break was caught by the matching test. Not covered, because they need a video, Tesseract or the API: speech, keyframe extraction end to end, OCR, the vision step and the pipeline runner.
8. **Documentation pass** and publishing of the public docs. *Done (2026-10-06).* README, MIT license and these notes are public.
9. **Phase 1 exit check.** *Done (2026-10-06).* `python -m src.verify` checks a processed lecture without any video, GPU or API, and it passes on all three lectures (decision 18). Phase 2 can open.

## Not covered by Phase 1

A clean slide-deck recording without annotations or a webcam has not been tried yet, and PDFs and PowerPoint files as course material are not processed. Neither blocks Phase 2.

## Target video types

Slides with a presenter, and code or screen demos. Whiteboard and blackboard recordings are out of scope for now.

## Phase 2 (in progress)

Done (2026-10-08): carrying the three versions of the slide text into the knowledge objects, chunking, the local embedding model with a cache, the Qdrant store with indexing as pipeline stage 10, dense search, cited answers, and an evaluation set (the questions themselves are private).

Next, in this order:

1. **Run the evaluation.** A command that runs every question through the search and reports hit@1, hit@3, hit@5 and mean reciprocal rank, and checks that the three questions the lectures do not cover are refused. This gives the first real numbers.
2. **Compare the choices that are only settings today:** which slide text is embedded (decision 15), how much slide text at all (decision 20), chunk size, and cutting at slide changes (decision 19).
3. **Keyword search merged with the meaning search** (BM25 and reciprocal rank fusion), kept only if hit@k improves.
4. **Reranking**, only if the measurements show it helps.
5. **The Gemini embedding provider and the comparison with bge-m3** (decision 12).
6. **Check that cited chunks support the claims**, by reading answers or with a judge.

Details and the design are in `phase2-plan.md`.

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
