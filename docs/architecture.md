# How the pipeline works

These are my notes on how Phase 1 works: what each stage does, how it does it, and why I built it that way. The reasoning behind the bigger choices, with numbers, is in [decisions.md](decisions.md). What is planned next is in [roadmap.md](roadmap.md) and [phase2-plan.md](phase2-plan.md).

Last updated: 2026-10-06 (Phase 1 complete).

## What it does

One lecture video goes in. What comes out is a list of **knowledge objects**, one per slide shown: what was on the slide (text, and a description of any diagram or code), everything said while it was on screen, and exact start and end times. For any moment in the video, you can tell what was said and what was on screen.

```
lecture video
   |-- 1 audio ----> 2 transcript ----------------------------.
   |                  (speech with timestamps)                  |
   |                                                            v
   '-- 3 keyframes -> 4 ocr -> 5 vision -> 6 clean ------> 7 align -> 8 knowledge objects
        (one picture   (text)   (describe    (remove         (which slide    (one record per slide,
         per slide)             figures,     junk text)       was showing?)   with its speech)
                                code, title)
```

Folder convention: the video goes in `data/raw/<lecture>/` (exactly one video file, otherwise the pipeline stops instead of guessing), and everything it produces goes in `data/processed/<lecture>/`. Both are ignored by git.

```
python -m src.pipeline lecture_01                      # one lecture
python -m src.pipeline --all                           # every lecture folder
python -m src.pipeline lecture_01 --force vision       # redo one stage (repeat the flag for several)
python -m src.pipeline lecture_01 --force ocr --force clean
```

Two rules apply to every stage:
- **Skip if the output exists**, unless forced. Transcription and frame extraction are slow, so this makes it practical to work on the later stages.
- **Build the whole output in memory, then write the file.** A crash can't leave a half-written file that the skip check would later trust.

Every stage reads files and writes files, so any stage's output is a JSON file I can open and read.

## Stage 1: audio (`src/ingestion/video_pipeline.py`)

`extract_audio` runs FFmpeg to pull the audio out of the video as a 16 kHz mono WAV file, which is the format Whisper wants. It calls FFmpeg through `subprocess` and does no audio work itself.

The WAV is only needed to make the transcript, and it is big (162 MB for an 88-minute lecture) but can be recreated in seconds. So the pipeline **skips this stage whenever `transcript.json` exists, and deletes `audio.wav` right after the transcript is written.** Forcing audio or transcript recreates it first. This is a config switch (`DELETE_AUDIO_AFTER_TRANSCRIPT`, on by default).

## Stage 2: transcript (`src/processing/speech.py`)

`transcribe` runs Faster-Whisper (the `small` model, on the GPU in float16) and writes `transcript.json`: a list of `{start, end, text}` segments. Every segment is checked against a schema (`TranscriptSegment`) before it is written.

Running on the GPU needs more than a GPU and a driver. It also needs the cuBLAS and cuDNN libraries, installed separately and on the PATH of the terminal session. The CPU mode worked without any of that and gave matching quality on a short test, so the GPU was a speed choice.

Speed: about 17 to 19 times real time (a 92-minute lecture in about 289 seconds).

## Stage 3: keyframes (`src/processing/keyframes.py`)

A lecture video has about 130,000 frames. I only want one picture per thing worth reading. `KeyframeExtractor` looks at the video every 5 seconds (it **seeks** to each time instead of decoding every frame, so an 88-minute video needs about 1,000 reads) and saves a frame when the screen has changed enough since the last *saved* frame.

`compute_difference` converts both frames to grayscale and returns the **fraction of pixels** that changed by more than `diff_threshold`. A new slide is saved when that fraction is above `change_area_threshold`, or when `max_gap_seconds` have passed with nothing saved.

**Why a fraction and not an average.** My first version averaged the pixel difference over the whole frame. The recording has a small webcam window that moves constantly, and it contaminated the average so much that no single threshold worked (25 keyframes, with a 21-minute gap). I considered cropping the webcam out by fixed pixel position and rejected it, because it would only fit that one layout. A fraction works wherever the overlay is: a small moving corner only ever changes a small share of the frame, while a real slide change changes a large share.

**Why the time-based safety net.** Live handwriting on a slide builds up over many seconds and may never cross the threshold in a single step, even though the change is real. Forcing a save after a long gap catches those. I checked that a fallback save really caught a meaningful moment (the presenter circling a formula), so it isn't hiding a weak detector.

**How I tuned it on the first lecture.** I printed the score at every checked frame and looked at real numbers. Raising the per-pixel threshold from 25 to 40 barely changed the plateau between transitions, which showed the plateau was real content (annotation) and not video noise. Lowering the area threshold from 0.10 to 0.05 dropped the share of saves coming from the safety net from about 59% to about 20%. The result was 64 keyframes for a 55-slide deck, and the extra ones are genuine revisits and progressive annotation states.

**The defaults are not universal.** They live in `config.py`, and a lecture can override them in its own `settings.json` (see "Per-lecture settings" below). A code screencast needed different values, because window switching and scrolling are large changes while typing is small: the defaults gave 156 keyframes for a 51-minute video, and `change_area_threshold` 0.15 with `max_gap_seconds` 90 gave 73.

## Stage 4: OCR (`src/processing/ocr.py`)

`extract_text` runs Tesseract on each keyframe and writes `visual_metadata.json`, one record per keyframe: `timestamp`, `image_path`, the raw `text` (one word per line), the average `confidence`, and a `word_confidences` list with one 0-to-1 number per word.

I chose Tesseract because it is classical OCR: nothing to host and no API cost. The weakness is that it reads everything on screen, so the application's menu bar lands in the text, handwriting becomes garbage words, and indentation in code is lost. Stages 5 and 6 deal with that.

`visual_metadata.json` is shared with the next two stages, which add fields to it. If OCR is redone with `--force ocr`, it keeps what the vision stage already stored, so redoing OCR costs no API calls.

## Stage 5: vision (`src/processing/vision.py`)

OCR can read words but not diagrams, plots or code layout. This stage sends every keyframe to Gemini (the `google-genai` library, free tier) and asks it to fill in a fixed form instead of writing free text:

| Field | Meaning |
|---|---|
| `title` | the slide's title, empty if none |
| `description` | for a figure: what it shows, including every label, axis title, annotation and caption. For code: the visible code with its indentation plus one line on what it does. Empty for a text-only frame. |
| `content_type` | `text_slide`, `title_slide`, `diagram`, `chart`, `code`, `equation` or `other` |
| `slide_number` | the page counter, only if the presentation program shows one |
| `clean_text` | all teaching text on screen, without menus or toolbars |

The reply is checked against a schema, so there is no sentence to string-match. `description` comes before `content_type` in the form on purpose: Gemini writes the fields in order, so it classifies after it has written what it sees. The prompt is generic: it says to ignore application interface and any small camera overlay, and it names no particular recording software. `description` empty means "nothing to describe", and `None` means "not done yet", so a rerun only retries what is missing.

**Speed is limited by the free-tier quota,** not by compute: 15 requests per minute per model. Extra parallel workers just got "429 quota exceeded" errors. So all requests share a limiter that spaces them evenly (14 per minute, a setting), with a small thread pool that only helps when the API is slow to answer. A 64-keyframe lecture takes about 5 minutes. Calls are retried up to four times with increasing waits, a failed keyframe is left as "not done" for the next run, and Ctrl+C saves the keyframes that already finished.

## Stage 6: clean (`src/processing/ocr_clean.py`)

This stage writes a new field, `cleaned_text`, and never changes the raw OCR text, so I can compare raw OCR, cleaned OCR and Gemini's `clean_text` fairly later. Two layers, because they catch different junk:

1. **Interface words.** Menu words such as `File Edit View Page Tools` are real words, so OCR reads them with *high* confidence, and a confidence filter can't see them. Instead I count how many keyframes each word appears on. Words on at least a quarter of the keyframes become candidates, and **one Gemini request per lecture** picks which candidates are software interface text. It can only choose from the candidates. The answer is saved as `junk_terms.json` so I can read it, and plain code then removes those words.
2. **Low confidence.** Garbage like `oP` or `sso` comes with a low confidence score, so words below a cutoff are dropped, and so are tokens with no letter or digit. The default cutoff is 0.3. Real code scores lower in Tesseract (`import` 0.49), so a code lecture overrides it to 0.1 in its settings file.

Measured against Gemini's own text on three lectures, cleaning raised the share of OCR words that Gemini also saw from 64% to 97% and from 71% to 92% on the two slide lectures, and from 46% to 58% on the code lecture. For code, Tesseract stays weak, and Gemini's text is the real source. The numbers and the limits are in [decision 11](decisions.md).

## Stage 7: align (`src/processing/alignment.py`)

The question: for this piece of speech, which slide was on screen? I sort the keyframes by time and use `bisect.bisect_right` on their timestamps, which finds the last keyframe at or before the segment's start in a binary search. Speech before the first keyframe gets `None`. The result is `alignment.json`.

**Limitation.** Only the segment's *start* time is used. A slide shown for a few seconds between two segment starts gets no speech. On the first lecture, 4 of 64 slides have an empty transcript, and I opened all four: three are quick section-title slides and one is a real content slide. They must not be dropped, because titles carry the topic. The plan is to merge blank slides into the next chunk in Phase 2.

## Stage 8: knowledge objects (`src/processing/knowledge.py`)

This merges `alignment.json` and `visual_metadata.json` into `knowledge_objects.json`: one object per keyframe with `lecture_id`, `start_timestamp`, `end_timestamp`, the joined `transcript`, `slide_text` (raw OCR), `slide_description`, `image_path`, `content_type`, `title` and `slide_number`. Fields are added only when a stage can fill them.

An illustrative object (made-up content, not from a real lecture):

```json
{
  "lecture_id": "lecture_01",
  "start_timestamp": 1160.0,
  "end_timestamp": 1285.0,
  "transcript": "So here we compare two models on the same test set ...",
  "slide_text": "Model comparison\nAccuracy ...",
  "slide_description": "A bar chart with two bars, one per model, ...",
  "image_path": "data/processed/lecture_01/keyframes/frame_1160.00.jpg",
  "content_type": "chart",
  "title": "Model comparison",
  "slide_number": 12
}
```

Details:
- A slide ends when the next one starts. The last slide ends when the last speech ends.
- Speech is grouped with one pass over the segments into a dictionary keyed by slide, and each slide then reads its own bucket. My first version looped over all segments once per slide. The output was identical, and at this scale the old loop would not have been slow, so I changed it for clarity, not speed.
- A missing description or content type becomes an empty string, so the schema validates.
- `lecture_id` is taken from the output folder name.
- Speech before the first slide is skipped.

**Check I use on every lecture:** each end time equals the next start time, and the word count in the knowledge objects equals the word count in `transcript.json`, so no speech is lost or duplicated.

## Per-lecture settings

Defaults live in `src/config.py`. A lecture can override them in an optional `data/raw/<lecture>/settings.json` next to its video. Anything left out keeps its default, and a misspelled section or setting name is an error, because a silent typo means you change a number and nothing happens. The file is read at the very start of a run, so a typo fails in a second and not after the slow transcription stage.

```json
{
    "keyframes": { "change_area_threshold": 0.15, "max_gap_seconds": 90 },
    "ocr_clean": { "min_word_confidence": 0.1 }
}
```

Changing a setting does nothing on a stage that already has output. Use `--force` for that stage and the ones after it.

## The pipeline runner (`src/pipeline.py`)

A thin conductor: it has no processing logic, it calls the stages in order with the right paths and times each one. With `--all`, each lecture runs inside a try/except so one bad video doesn't lose the rest. Because every stage skips finished work, running a finished lecture again takes about no time.

Known gap: forcing an early stage does not automatically redo later ones. After `--force vision`, the knowledge objects need `--force knowledge` too.

I decided against Celery: it only runs more jobs at once, the slow stage is bound by one GPU, the API stage is a network wait that a thread pool handles, and it is poorly supported on Windows. Reasoning in [decision 8](decisions.md).

## The exit check (`src/verify.py`)

`python -m src.verify lecture_01` (or `--all`) checks that a processed lecture is complete and consistent. It reads only the files in `data/processed/<lecture>/`, so it needs no video, GPU or API key, and it never changes anything. It checks that every file exists and has the right shape, the vision and cleanup stages finished for every keyframe, each slide ends exactly when the next one starts, the word count in the knowledge objects equals the word count in the transcript, and every piece of speech lies on exactly one slide that `alignment.json` agrees with. Anything that is not broken but worth a look (slides nobody spoke during, speech before the first slide) is a warning, not a failure.

`python -m src.verify lecture_01 --at 1420` shows what Phase 1 promises: the slide on screen at that second (type, title, slide number, image, description) and the speech around it.

It passes on all three lectures. It checks completeness and consistency, not quality: it cannot tell whether Whisper heard a word correctly or whether a description is right. Results and the way I checked that it can fail are in [decision 18](decisions.md).

## Tests

`python -m pytest` runs 65 tests in about 3 seconds, with no video and no API. They cover `compute_difference`, `align`, the grouping in `build_knowledge_objects`, the OCR cleaning functions, the settings loader and the exit check, using tiny hand-made data. A fake Gemini key is set before anything is imported, so a test can never spend quota. To check that the tests can fail, I broke three things on purpose (an off-by-one in `align`, a cutoff comparison, a missing `.strip()`), and each break was caught by the matching test. Speech, keyframe extraction end to end, OCR, the vision stage and the runner are not covered by the tests, because they need a video, Tesseract or the API. The exit check covers their output on real lectures instead.

## Results on three lectures

| | Length | Keyframes | Words in transcript = knowledge objects | Total time |
|---|---|---|---|---|
| Lecture 1: annotated slides, webcam overlay | 88 min | 64 | 11,194 | n/a (processed in steps) |
| Lecture 2: same style as 1 | 92 min | 67 | 11,192 | 691 s |
| Lecture 3: public code screencast (terminal and editor), presenter webcam and room-camera cuts | 51 min | 73 | 10,051 | 590 s |

In all three, each slide's end time equals the next slide's start time and no speech is lost. Speech recognition and the vision stage together take about 85% of the time.

## Known limitations

- Whisper is set to run on an NVIDIA GPU. Running without one needs a change in `speech.py` (`device="cpu"`, `compute_type="int8"`).
- Tesseract is weak on terminal and code text, so for code the vision model's text is the useful source.
- Interface words are removed everywhere in `cleaned_text`, so a real use of "file" or "view" in lecture content is removed too. The raw text keeps it.
- Slides shown for only a few seconds get no speech (stage 7).
- Only lecture video is processed so far. PDFs and PowerPoint files are planned.
- The free Gemini quota sets the speed of the vision stage.
