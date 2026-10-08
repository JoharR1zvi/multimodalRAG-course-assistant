# Decision log

One entry per decision: the problem, what I considered, what I chose, why, and what happened. Newest decisions go at the bottom. Entries marked *pending* are waiting on results.

Last updated: 2026-10-08.

---

## 1. Speech to text: Faster-Whisper on the GPU

**Problem.** Turn an 88-minute lecture into text with timestamps, because every later citation depends on them.

**Options.** The original Whisper package, WhisperX, a hosted transcription API, Faster-Whisper on CPU, Faster-Whisper on GPU.

**Chosen.** Faster-Whisper, `small` model, on the GPU (RTX 4050, 6 GB) in float16.

**Why.** It runs locally (free, private), returns timestamped segments, and is much faster than the original package. The CPU version worked out of the box and gave matching quality on a one-minute test, so the GPU was a speed choice, not an accuracy one.

**What I learned.** Having a GPU and driver is not enough. The GPU path also needed two CUDA runtime libraries (cuBLAS and cuDNN) installed separately and added to the PATH for each terminal session.

**Result.** About 690 segments for the full lecture, text quality confirmed by spot checks.

---

## 2. Detecting slide changes

**Problem.** Pick the moments worth keeping a picture of, without storing 130,000 frames.

**Options.** (a) Average pixel difference over the whole frame with one threshold. (b) Crop out the webcam overlay by fixed pixel coordinates, then average. (c) Measure the *fraction of pixels* that changed a lot, plus a time-based safety net.

**Chosen.** (c).

**Why.** (a) produced only 25 keyframes with a 21-minute gap, because the small moving webcam window contaminated the average. (b) would have worked for this one recording and broken on any other layout, so I rejected it. (c) works wherever the overlay is, because a small moving corner only ever changes a small share of the frame, while a real slide change changes a large share. The safety net (force a save after 180 seconds with no detected change) exists because live handwriting on the slides builds up gradually and never crosses the threshold in a single step. I checked that a fallback save really did catch a meaningful moment (the presenter circling a formula), so it is not just hiding a weak detector.

**How it was tuned.** I printed the score at every checked frame and looked at real data instead of guessing. Raising the per-pixel threshold from 25 to 40 barely changed the plateau between transitions, which showed the plateau was real content (annotation) and not noise. Lowering the area threshold from 0.10 to 0.05 dropped the share of saves coming from the safety net from about 59% to about 20%.

**Result.** 64 keyframes for a 55-slide deck. The extra ones are genuine revisits and progressive annotation states, confirmed by looking at the images.

**Caveat.** These numbers are tuned to one recording style. They are moving into config so they can be changed per lecture.

---

## 3. Reading slide text: Tesseract

**Problem.** Get the text off each keyframe.

**Options.** Tesseract, PaddleOCR, or sending every frame to a vision model.

**Chosen.** Tesseract.

**Why.** Classical OCR: nothing to host, no API cost, and it follows the rule of using a traditional technique before reaching for a model.

**Result and weaknesses.** It works, but it reads everything on screen, so the application menu bar and tooltips end up in the text, along with garbage tokens from handwriting. It also loses indentation, which will matter for code lectures. Cleanup approach is decision 11.

---

## 4. Describing diagrams: a vision model for figures only

**Problem.** OCR can read words but not charts, plots or diagrams.

**Chosen.** Send each keyframe to Gemini (free API) and ask it to describe only the diagram, or reply with an exact fixed phrase if there is none. That phrase is stored as an empty string, so the data distinguishes "not described yet" (`null`), "no diagram" (empty) and "described" (text).

**Why.** It answers exactly what OCR cannot, at the cost of one call per keyframe. Calls are retried up to four times with increasing waits, because free-tier rate limits are real.

**Result.** 27 of 64 keyframes have a description.

**Known issue.** The prompt currently names the recording tool's interface. That has to be generalized (decision 9).

**Update (2026-10-06).** Fixed. The prompt is generic and the reply is a fixed form, see decision 14.

---

## 5. Data shapes and stage rules

**Problem.** Seven stages that read and write each other's files need to stay debuggable.

**Chosen.** Pydantic models for every data shape, and two rules for every stage. First, skip the work if the output already exists unless forced (transcription and frame extraction are slow, so this makes iterating on later stages practical). Second, build the complete output in memory and only then open the output file, so a crash cannot leave a half-written file that the skip check would later trust.

**Why.** Bad data fails at the stage that produced it instead of three stages later, and every stage output is a JSON file I can open and read.

---

## 6. Matching speech to slides

**Problem.** For each piece of speech, which slide was on screen?

**Chosen.** Sort keyframes by time and use a binary search (`bisect`) to find the last keyframe at or before the segment's start time.

**Limitation found later.** Only the start time is used. A slide shown for a few seconds between two segment starts receives no speech. Four of 64 slides in the first lecture have empty transcripts. I opened all four: three were section title slides ("Calibration scores" and similar) clicked through quickly, and one was a real content slide with annotations. They must not be dropped, because titles carry the topic. The plan is to merge blank slides into the following chunk (see `phase2-plan.md`).

---

## 7. One knowledge object per slide

**Problem.** Combine slide text, diagram description and speech into one record.

**Chosen.** One object per keyframe: speech grouped by slide, start and end time, slide text, description, image path. The schema is deliberately minimal and gains fields only when a stage can fill them.

**Implementation note.** The first version looped over all segments once per slide (slides times segments comparisons). I changed it to one pass that drops each segment into a dictionary keyed by its slide, then a lookup per slide. The output is identical. At this scale the old loop would not have been slow, so the reason was clarity and scaling, not speed.

**Verification.** 64 objects; each end time equals the next start time; the word count of the knowledge objects equals the word count of the transcript, so nothing was lost or duplicated.

---

## 8. Orchestration: a plain sequential runner, no task queue

**Problem.** Processing one lecture by hand meant seven commands with long paths. Processing 12 or 13 lectures would be far worse, and I wanted per-stage timing.

**Chosen.** A thin `pipeline.py` that calls the stages in order, times each one, accepts `--force` per stage, and with `--all` continues past a failed lecture.

**Celery rejected.** A task queue only runs more jobs at once. The slow stage (speech recognition) is bound by a single 6 GB GPU, so extra workers would compete for memory, not go faster. The API calls are network waits that a small thread pool handles without a message broker, Celery is poorly supported on Windows, and processing the course is a one-time batch. A queue would make sense later for a web upload flow, and even then simpler options come first.

**Open detail.** The vision stage edits `visual_metadata.json` in place, so forcing OCR wipes its descriptions and repeats all the API calls. The pipeline prints a warning.

---

## 9. The pipeline must not be tied to one recording style

**Problem.** Everything so far was tuned on one lecture recorded with annotation software and a webcam overlay. The goal is that someone can upload a different lecture and it works.

**Chosen.** Target video types are slides with a presenter and code or screen demos. Whiteboard recordings are out of scope. No fixed pixel crops, anywhere. Layout-specific wording leaves the vision prompt, thresholds become configuration, and the second test lecture is chosen to look different on purpose.

**Expected difficulty.** Code demos change gradually (typing and scrolling change a few percent of the screen), so the slide-change threshold will under-detect them.

---

## 10. Phase 2 design choices (planned)

Full reasoning in `phase2-plan.md`. Summary:

- **Chunks are built from timed speech segments, not from the joined slide transcript**, so citations can point to a precise moment.
- **Target about 350 words, merging consecutive slides.** With the first lecture's numbers (median slide 162 words, only 8 of 64 over 350), chunking strictly within slides would make most chunks identical to slides.
- **Slide text and diagram descriptions are embedded together with the speech.**
- **Embeddings: implement a local model (bge-m3) and an API model behind one swappable interface, then compare them** on an evaluation set.
- **Qdrant in embedded mode**, behind a small interface.
- **Measure before tuning:** write a question set with known answer locations first, then compare dense search, hybrid search (dense plus BM25, merged by reciprocal rank fusion) and optional reranking by hit@k.
- **A hosted database would not save disk.** Vectors for a full course are a few MB. The disk is taken by the audio file and the videos, so the audio file is deleted after transcription.

**Status (2026-10-08).** Built, except the embedding comparison, hybrid search and reranking. What happened to each point is in decisions 19 to 23.

---

## 11. Cleaning interface text out of OCR without hardcoding the layout

**Problem.** Menu-bar text and tooltips pollute the slide text, and a fixed crop would only fit one recording.

**Considered.** Fixed crop (rejected). A plain frequency filter (words on most slides are interface). An LLM pass per frame. One LLM pass per lecture over frequency statistics.

**Chosen direction.** Layered: (1) a generic vision prompt that returns clean slide text; (2) a once-per-lecture step that counts how many slides each word appears on and asks an LLM to classify the most frequent words as interface or content, saved as an inspectable list and applied with plain code; (3) a per-word OCR confidence filter for random garbage; (4) only if needed, ignore words in screen regions that never change across the video.

**Why the combination.** Frequency finds candidates, but cannot tell a real content word that happens to be frequent from an interface word. A model alone reads context-free strings. Together, frequency proposes and the model judges, for the cost of one small request per lecture. It also catches OCR-corrupted menu words that an exact-match frequency count treats as different words.

**Status.** Built (2026-10-06). Layer 1 is the vision prompt (decision 14), layers 2 and 3 are a new pipeline stage, and layer 4 turned out to be unnecessary so far.

**What I built.** A new `clean` stage writes a *new* field, `cleaned_text`, and never touches the raw OCR text, so raw OCR, cleaned OCR and the vision model's text can be compared fairly later (decision 15).
- *Interface words.* Words that appear on at least 25% of a lecture's keyframes (and at least 5) become candidates. One request per lecture asks the model which candidates are software interface text. It may only pick from the candidates, and its answer is saved as `junk_terms.json` next to the other outputs, so I can open it and check it. Plain code then removes those words. Menu words are read by OCR with *high* confidence (they are real words), so a confidence filter cannot catch them, which is why this layer exists.
- *Confidence and symbols.* Words below a confidence cutoff are dropped, and so are tokens with no letter or digit. Garbage like `oP` or `sso` scores 0 to 0.3 while real slide words score 0.9 or more, so this layer catches what the first one cannot.
- To make the second layer possible, OCR now saves one confidence per word. Redoing OCR also keeps the vision results already in the file (before, `--force ocr` wiped them and repeated every API call).

**What it found.** Interface words: the first two lectures `page options help file tools view edit layer` plus the OCR misreading `xournat` of the app's own name; the code lecture `file window help`. The model correctly kept ordinary frequent words like `the`, `model` and `metrics`.

**Results**, using the vision model's text as a rough yardstick (median per slide):

| | OCR words the model also saw, raw | after cleaning | real words that survive cleaning (median / worst slide) |
|---|---|---|---|
| Lecture 1 (slides) | 64% | 97% | 100% / 77% |
| Lecture 2 (slides) | 71% | 92% | 100% / 92% |
| Lecture 3 (code) | 46% | 58% | 99% / 50% |

**The cutoff is not one number.** I swept it from 0 to 0.6. On slide lectures 0.3 to 0.5 gives 92 to 100% agreement with nearly all real words kept. On the code lecture, real code scores low in Tesseract (`import` 0.49, `hello.py` 0.53), and a cutoff of 0.3 or more deleted words like `python`, `hello` and `codelab`; about 0.1 keeps nearly everything. So the default is a conservative 0.3, and the code lecture overrides it to 0.1 in its settings file, which is what that file is for.

**Limits.** Interface words are removed everywhere, so a genuine use of "file" or "view" in lecture content is also removed from the cleaned text (the raw text keeps it). Tesseract stays weak on terminal text, so for code the vision model's text is the real source. Short garbage with moderate confidence (`mH`, `Gl`, `ax`) and fragments of URLs in slide footers are what is left over.

---

## 12. Embedding model comparison (pending)

To be filled in after steps 2.2 and 2.4: hit@k, latency, cost and rate-limit behaviour for the local model against the API model, and which one became the default and why.

**Status (2026-10-08).** Only the local model (bge-m3) is built and working, so the comparison has not been run. The Gemini provider answers "not built yet". What I learned about the local model so far is in decision 20.

---

## 13. Keyframe settings: defaults in config, overrides in a per-lecture file

**Problem.** The slide-change numbers were tuned by eye on one recording style. A code screencast changes only a few percent of the screen at a time, so the same numbers would miss most changes there, and a clean slide deck may need different ones again. But editing the shared numbers for one lecture silently changes what an earlier lecture would produce if I re-ran it.

**Options.** (a) Edit the defaults for each new lecture. (b) Command-line flags per run. (c) An optional `settings.json` next to each lecture's video.

**Chosen.** (c). The defaults moved into `config.py` with the same values. The file lists only the numbers that differ for that lecture, and anything left out keeps its default. A misspelled key raises an error instead of being ignored, because a silent typo means you change a number, nothing happens, and you do not know why. The file is read at the very start of a run, so a typo fails in a second and not after the slow transcription stage.

**Why.** Each lecture's settings live next to its video and are written down, so any lecture can be reproduced exactly and the shared defaults never move. Flags would have to be retyped every run, and `--all` could not use different values per lecture.

**Verification.** Re-extracting keyframes for the first lecture into a scratch folder gave the same 64 filenames as the original run. That also confirmed that an earlier readability rewrite of the extractor behaves the same. No file means defaults, a typo gives an immediate error, and a valid override is applied.

**Limits and what comes next.** A hand-written JSON file is a developer tool, not something a user should have to touch. The plan is an automatic choice that fills in the same numbers, with the file kept as the final override: (1) look at a few sampled frames, decide "slides" or "code demo", and use a preset; (2) measure the video's normal frame-to-frame noise and set the threshold above it; (3) later, a "what kind of video is this" choice in an upload form. I will build this after the code-screencast lecture shows what the numbers need to be, because designing it now would be guessing. Each run should also save the settings it actually used next to its outputs.

**First real use (2026-10-06).** The code screencast (decision 17) needed its own settings, and they were not what I predicted. I expected the default 5% change threshold to miss most changes, because typing changes only a few percent of the screen. Instead the defaults saved 156 keyframes for a 51-minute video, far too many, because switching between terminal and editor windows and scrolling output are large changes. A higher threshold alone then caused the opposite problem: stretches of pure typing were only caught by the safety net. The settings I settled on were a change threshold of 0.15 and a safety-net gap of 90 seconds, which gave 73 keyframes. The point for the later automation: one global threshold cannot separate "window switch" from "typing", so a preset per video type, or a smarter detector for code, is the realistic direction.

---

## 14. Vision step: generic prompt, a fixed reply form, and the real speed limit

**Problem.** The prompt named one recording tool's toolbar and webcam. It only looked for diagrams, so a code frame would have come back as "no diagram" and the code would be lost. And "no diagram" was detected by comparing the reply with an exact sentence, which breaks if the wording shifts.

**Chosen.** A generic prompt plus a form the model must fill in: `title`, `description` (figures and code only), `content_type` (text slide, title slide, diagram, chart, code, equation, other), `slide_number` if a page counter is visible, and `clean_text` (the teaching text without menus). The reply is checked against a schema, so there is no sentence to match. Temperature is 0 so reruns differ as little as possible.

**What the first version got wrong** (checked on a copy of the metadata, never on the real file):
- `content_type` was filled in before the description, so slides with a figure plus text were labelled plain text slides: 11 diagram labels against 27 diagrams found before.
- Descriptions leaked onto text-only slides: 9 of them got a summary of their bullet points.
- Once I told it any figure counts, it read a small coloured difficulty badge as a diagram.

Fixes: `description` now comes before `content_type` in the form, so the model writes what it sees before classifying; the prompt says decorative items are not figures and formulas belong in `clean_text`.

**Result after the fixes.** A 29-slide targeted check passed 29 of 29. On the full lecture, all 27 earlier diagram slides are still described with none lost, no keyframe failed, the slide number was found on 64 of 64, and a title on 62 of 64. Descriptions got shorter (median 478 characters, was 1,188) but keep the labels and the concept. Three slides that are screenshots of quoted text now get a description. Section-title slides are labelled "text slide", but their `title` field is correct, which is what I need.

**Speed: the limit was a quota, not slowness.** The first full run took 702 seconds. The free tier allows 15 requests per minute per model. Extra parallel workers just got "429 quota exceeded" errors, and even one worker at about 3 seconds per call is over the limit. The fix is a shared limiter that spaces requests evenly (14 per minute), plus a small thread pool that only helps when the API answers slowly. The full lecture now takes about 304 seconds, close to the floor of about 4.3 seconds per keyframe on the free tier. Going faster needs either a higher quota (a setting in `.env`) or several frames per request. Lowest thinking level was about a second faster per call with identical output on the slides I tested.

**Smaller changes.** A failed reply now logs its finish reason and the start of the reply. Ctrl+C saves the keyframes that already finished instead of losing them.

**Applied to the real first lecture (2026-10-06).** One more prompt line asks the model to name every label, axis title, legend entry, annotation and caption in a figure, because comparing a few old and new descriptions showed the shorter version kept axis names but dropped small things like a source caption and an annotated value. The real run (298 seconds) kept all 27 earlier diagram slides, failed on none, found a slide number on 64 of 64 and a title on 62, and no text-only slide carries a description. Median description length is 532 characters. The old versions of the two files are kept in a backup folder inside the lecture's output folder.

**Known small issue.** Camera shots of the presenter (the code lecture cuts to the room now and then) are labelled "text slide" with an empty title and empty text, not "other". They carry about 4% of the spoken words, so they must stay as keyframes, but the label is wrong. A prompt line about camera shots would fix it.

---

## 15. Which slide text to use: raw OCR, cleaned OCR, or the vision model's text (pending)

**Problem.** There are now two sources of slide text: Tesseract and the vision model's `clean_text`. Comparing raw OCR to the model's text is not a fair fight, because raw OCR carries the interface junk.

**Early numbers** (63 slides, word level only): a median 5% of the model's words do not appear in the OCR text (max 29%), mostly words OCR missed. A median 36% of the OCR words are missing from the model's text, and the examples are interface words (`file`, `edit`, `view`, `help`, `layer`) and garbage like `qaqar`. This cannot tell junk from real omissions, and it says nothing about meaning.

**Plan.** Finish the OCR cleanup layers (decision 11) first. Build the evaluation set. Then run retrieval three times, changing only the text that gets embedded: (1) raw OCR, as the baseline that shows what cleanup buys; (2) cleaned OCR; (3) the model's `clean_text`. Compare hit@k.

**Things to weigh.** OCR is literal, local, free and works offline. The model's text is much cleaner and keeps code indentation, but it can paraphrase, and it depends on a quota and on the service being up. A likely outcome is the model's text as the main source with OCR as the fallback for a frame that fails. The measured numbers decide it, and the code screencast is the hardest test.

**After cleaning the OCR (2026-10-06).** On the slide lectures, 92 to 97% of the cleaned OCR words are also in the model's text, so cleaned OCR is now close to it for plain slide text (decision 11). What the model adds beyond that is diagrams and formulas, which OCR cannot read. Of the three versions, the comparison at retrieval time still decides.

**Status (2026-10-08).** All three versions now travel with every chunk (`slide_text`, `cleaned_text`, `clean_text`). The version that gets embedded is a setting, and the default is the vision model's text, because it is the cleanest in the samples I read. The retrieval comparison itself has not been run. The evaluation command exists now, and the baseline with the default text is in decision 24. Decision 20 adds a second question to it: how much slide text to embed at all.

**What the code lecture already shows.** On a frame of a terminal session, the model's output is the terminal text line by line with indentation, and Tesseract's output for the same frame is a jumble (`Terminal”|File.|Edit:|Scrolibach)|...|npartante`). On that lecture the share of OCR words that the model's text lacks has a median of 62%, against 29-36% on the slide lectures. For code, the model's text is clearly the better source, so the open question is mostly about slide lectures.

---

## 16. Deleting audio.wav after transcription

**Problem.** The extracted audio exists only to feed speech recognition, but it is large (162 MB for the 88-minute lecture, 95 MB for the 51-minute one), and it can be recreated with one FFmpeg command in a few seconds (8.6 s for a 92-minute lecture).

**Chosen.** The pipeline deletes it as soon as `transcript.json` exists. This is a config switch, on by default (`DELETE_AUDIO_AFTER_TRANSCRIPT` in `.env`). The audio stage is skipped whenever the transcript already exists, unless audio or transcript is forced, in which case the audio is recreated first.

**Result.** Verified on the first lecture (stage skipped, 162 MB deleted). With the audio gone, the processed data for three lectures is 28.5 MB in total.

---

## 17. Three lectures, three recording styles: results and timings

**The three lectures.** Lecture 1: annotated slides with a webcam overlay, 88 minutes. Lecture 2: another recording of the same course in the same style, 92 minutes, so it checks consistency and not generalization. Lecture 3: a public Creative Commons recording of a programming class, 51 minutes, 640x480, a terminal and a text editor with live-typed code, a presenter webcam in the corner, and cuts to a full-screen camera shot of the room. This is the real generalization test.

**Results (all with the same code).**
- Lecture 2 on default settings: 67 keyframes, speech word count equal in the transcript and the knowledge objects (11,192), end times chain correctly, a slide number on 67 of 67, a title on 65. The same seven interface words (`Page Help View Options File Tools Layer`) appear in the OCR text of at least half the slides, which is the junk the cleanup layers target.
- Lecture 3 with its settings file (decision 13): 73 keyframes, 10,051 words in both transcript and knowledge objects, end times chain correctly, 46 frames labelled code, 4 diagram, 21 text (8 of them camera shots). No slide numbers, which is right because none are shown on screen.

**Timings (seconds).**

| | audio | transcript | keyframes | OCR | vision | total |
|---|---|---|---|---|---|---|
| Lecture 2 (92 min, 67 keyframes) | 8.6 | 288.8 | 38.3 | 54.9 | 300.3 | 691 |
| Lecture 3 (51 min, 73 keyframes) | 4.5 | 178.3 | 33.8 | 46.4 | 327.0 | 590 |

Speech recognition runs at about 17 to 19 times real time on the GPU. The vision step is 43% to 55% of the total, and it is limited by the free-tier quota, not by compute.

**What that means for the speed work.** The biggest lever is the Gemini quota, which is a setting, because the free tier floors vision at about 4.3 seconds per keyframe. The next is overlapping the GPU stage with the keyframe, OCR and vision stages, which I estimate would save around 40% per lecture (an estimate, not measured). Spreading OCR over CPU cores would save under a minute per lecture, so I am skipping it. Processing 12 or 13 lectures one after another would take roughly two to two and a half hours today.

---

## 18. The Phase 1 exit check

**Problem.** Phase 1 promises that for any moment in a lecture you can tell what was said and which slide was on screen. Until now I had checked that by hand, lecture by lecture, with one-off scripts. I wanted one command that does it the same way every time.

**Chosen.** `python -m src.verify <lecture>` (or `--all`). It reads only the processed files, so it needs no video, no GPU and no API key, and it never changes anything. It reports each check as PASS, FAIL, WARN or INFO, and the exit code is 1 if anything failed.

**What it checks.**
- All four files exist and every item has the right shape, and the keyframe images exist and match the metadata.
- The vision and cleanup stages finished for every keyframe.
- There is one knowledge object per keyframe, each names its lecture, and each starts at its keyframe.
- Each slide ends exactly when the next one starts, with no gaps and no overlaps.
- The word count in the knowledge objects equals the word count in the transcript, so no speech is lost or duplicated.
- The promise itself: every piece of speech falls inside exactly one slide's time span, and `alignment.json` picked that same slide. This is a second, independent calculation, not a copy of the one in `align`.
- Warnings that are not failures: slides nobody spoke during (kept on purpose), and speech before the first keyframe.

`python -m src.verify <lecture> --at 1420` shows the promise in action: it prints the slide on screen at that second (type, title, slide number, image, description) and the speech around it.

**Results (2026-10-06).** All three lectures pass: 64, 67 and 73 knowledge objects, with 11,194, 11,192 and 10,051 words matching exactly. The only warnings are the slides with no speech (4, 2 and 5 of them).

**Does it catch problems?** I copied the first lecture's processed folder six times and broke each copy differently: one slide's speech deleted, one slide ending a second late, one image file deleted, one description missing, one segment matched to the wrong slide, and one keyframe without cleaned text. Each was caught by exactly the check that should catch it, and the untouched lecture still passed. Twenty-two automated tests cover the same cases on tiny hand-made data. One of my own tests failed at first because I had made inconsistent fake data, and the check flagged it.

**What it does not prove.** It checks completeness and consistency, not quality. It cannot tell me whether Whisper heard a word correctly, whether the vision model's description is right, or whether Tesseract's text is good. Those were judged separately, by reading samples and comparing against the vision model's text (decisions 11, 14 and 15).

---

## 19. Chunking: cut by the length of the speech, carry the slide text along

**Problem.** The knowledge objects are one per slide, but slides are very uneven. In the first lecture the median slide has 162 words of speech, 26 of 64 slides have under 100 words, 8 have over 350, and 4 have none. A good thing to search is a piece of about the same size that points at a precise moment.

**Considered.**
- *One chunk per slide.* Rejected for the numbers above: tiny slides make useless search targets and long ones make blurry ones. It also breaks when one slide is shown in several screen states, which happens a lot with annotated slides. The next screen change kept the same title in 12 of 59 cases in lecture 1 and in 27 of 62 in lecture 2.
- *Cut by length and attach the slides afterwards.* Chosen.
- *Cut where the meaning shifts* (semantic chunking). It costs more and tends not to beat simple methods by much, so I left it for later.
- *Cut by length but prefer a real slide change.* Not built yet, see the experiment below.

**Chosen.** Pieces of speech go into a bucket in time order. At about 350 words the bucket is sealed, and the next one starts with a copy of the last piece (one piece of overlap). A bucket never goes over 450 words. A slide nobody spoke during waits and joins the next bucket, because a title comes before the content it names. A tiny last bucket is merged into the one before it. Chunks come from the timed speech pieces in `alignment.json`, not from the joined text, so every chunk has exact start and end times. Each chunk keeps all three versions of the slide text and the diagram description, and `embed_text` is built from slide text, description and speech. Sizes are counted in words of speech, which turned out to matter, see decision 20.

**Results (2026-10-08).** 33, 33 and 30 chunks for the three lectures, with a median of 356, 353 and 356 words of speech and about three slides per chunk. The exit check now verifies chunks as well (decision 18 describes the rest of the check). All three lectures pass. Eight new tests each damage correct chunks in one way and check that the matching check fails: a chunk over the maximum, a slide in no chunk, lost speech, a chunk reaching past its slides, duplicate ids, a broken neighbour link, and no chunks at all.

**One check failed on real data, and the check was wrong.** "Chunk times lie inside the slides it lists" failed on all three lectures. A speech piece belongs to the slide showing when it starts, so a piece that starts just before a slide change can end up to 7.4 seconds after it, and a chunk that ends on that piece inherits the overrun. The chunker was right. I allowed 10 seconds of overrun in the check, and a test still catches a chunk that ends minutes late.

**What I measured about the cuts.** Because cuts ignore slides, they almost always fall in the middle of a slide: 31 of 32 cuts in lecture 1, 30 of 32 in lecture 2 and 22 of 29 in lecture 3. By hand I also saw a search hit whose speech preview was about one topic while the matching part was later in the chunk, which suggests chunks sometimes mix topics.

**Experiment I will run later.** Once a bucket has at least about 250 words, close it at the next real slide change (a changed title, not just a new keyframe), and use the current rule when there are no titles (the code lecture has titles on only 12 of 73 slides). I will compare it with the current rule by hit@k on the evaluation set (decision 23) and keep whichever finds the right chunk more often. Chunk size will be tested the same way.

---

## 20. Embeddings: bge-m3 on the GPU, and the input limit I missed

**Problem.** Turn chunks and questions into vectors, with a model I can swap.

**Chosen.** BAAI/bge-m3 through sentence-transformers, on the GPU in half precision, giving 1,024 numbers per text, scaled to length 1. The provider and model come from `config.py`. The vector size is measured and never typed in. Only the local provider is built; the Gemini one answers "not built yet", so the comparison in decision 12 is still pending. Every vector is cached on disk, in a file named by a hash of the model, the token limit and the text.

**A mistake I found by measuring.** I had sized chunks in words, but the model reads tokens (pieces of words) and cuts everything after 1,024 tokens without a warning. I counted the tokens of all 96 embedded texts:

| Lecture | Chunks | Over 1,024 tokens | Longest |
|---|---|---|---|
| Lecture 1 | 33 | 16 | 1,979 |
| Lecture 2 | 33 | 18 | 1,845 |
| Lecture 3 | 30 | 21 | 2,549 |

So 55 of 96 chunks lost the end of their text. The embedded text starts with the slide text and the description and ends with the speech, so it was the speech that was cut. The speech alone was never the problem: it is about 450 to 475 tokens per chunk (at most 527). The rest is slide material. The slide text is about 320 to 384 tokens per chunk and the description 171 to 422, and slide text costs 1.9 to 2.2 tokens per word against 1.3 for speech, because of symbols and web addresses.

**Fix.** The limit is now 3,072 tokens (the model accepts up to 8,192), the batch size went from 8 to 4 for a 6 GB card, and the service prints a warning if any text is still cut. The token limit is part of the cache key, because otherwise the 96 vectors already saved from cut-off text would have been reused silently. No text is cut now, and embedding all 96 chunks takes about two seconds per lecture once the model is loaded (peak GPU memory about 2.7 GB). A quick check by hand gave the same top results before and after the change.

**What this leaves open.** More than half of what is embedded is slide material, so the vector may lean toward the slide and away from the speech. Whether that helps or hurts is a measurement. The comparison I plan: speech only, speech plus slide text, and everything, with the three versions of the slide text (decision 15). I would trim repeated slide text and cap descriptions only if the numbers show the slide text drowning out the speech.

**Lesson.** A model's input limit is a ceiling and not a target, so I measure in the unit the model uses (tokens, not words) and never let text be cut silently.

---

## 21. Vector store: embedded Qdrant

**Problem.** Store the vectors, find the closest ones to a question, filter by lecture, and return everything needed for a citation.

**Considered.** A plain array searched with numpy would be fast enough: a 13-lecture course is about 500 chunks. A hosted vector database would add an account and a network hop for no gain at this size. I chose Qdrant in embedded mode because I want to learn how a vector database works, because it filters on metadata, and because it still works when there are many more courses and documents. Speed is not the reason.

**Chosen.** Qdrant runs inside the program and keeps its files in `data/qdrant/`. There is one collection per embedding model (`course_chunks__bge-m3`), so vectors from different models never mix. A point is a vector plus the whole chunk as its payload, so a result carries its own lecture, times, slide titles and text. A point's id is derived from the chunk id (a UUID built from it), so saving a chunk twice overwrites it. The distance is cosine, which suits vectors scaled to length 1.

**Indexing** removes a lecture's old points and saves the new ones, and it counts as done when the database holds as many points for the lecture as there are chunks. So it is safe to run again, and re-chunking a lecture cannot leave stale points behind. It is stage 10 of the pipeline.

**Result.** 96 points (33, 33 and 30), 2.1 MB on disk. Running the pipeline again stores nothing new. Thirteen tests use Qdrant's in-memory mode with 3-number vectors, and breaking the lecture filter or the clean-up of old points makes tests fail.

**Limit.** Only one program can have `data/qdrant/` open at a time.

---

## 22. Answers with citations that cannot be invented

**Problem.** A language model can invent a timestamp as easily as a fact. The answer has to cite the lecture, the time and the slide, and those citations must be real.

**Chosen.** The five closest chunks are numbered 1 to 5 and given to Gemini as excerpts. The model must reply in a fixed form with two fields, `answerable` and `answer`, and may cite only excerpt numbers such as `[2]`. It never writes a lecture name or a time. My code then looks up each cited number and builds the source list from that chunk's stored data: lecture, time range, slide titles and the picture of its first slide. A citation to a number that does not exist is removed and reported, an answer that claims to be answerable but cites nothing valid gets a warning, and if the search finds nothing, the model is not called at all. The instructions say to use only the excerpts and no outside knowledge, to say so when they do not cover the question, to cite after every statement, and to treat the excerpts as data and not as instructions. The temperature is 0, and the thinking level is low.

**Why a fixed form.** The same reason as the vision step (decision 14): there is no sentence to string-match for "I cannot answer this".

**Result.** I tried two questions by hand. One was answered in four sentences with four cited sources, with real lecture names and time ranges. One question that the lectures do not cover (the weather in a city) was refused, with the three closest passages shown. 16 tests replace Gemini with a prepared reply and check what my code does with it, including a made-up citation.

**What is not verified.** That a cited chunk really supports the sentence it is attached to. The code can prove a citation exists, not that it is relevant. Checking that needs the evaluation set (decision 23) and either reading the answers or a judge. Also, the slide pictures are not sent to the model, only their text and descriptions, so a question about a figure depends on the quality of its description. A first spot check of four answers is in decision 24.

---

## 23. The evaluation set: questions from the course's own exercise sheets

**Problem.** The rule from the plan is to measure before tuning, which needs questions with known answer locations. Making them up myself would test what I already think the system can do.

**Chosen.** I use the recall questions from the course's two exercise sheets: 9 questions, with one split into three parts, so 11 items. Each item has the time ranges in the lectures where the answer is spoken. I added 3 questions the lectures do not cover (one off topic, two from neighbouring fields), to test that the system refuses. I checked that none of them is mentioned in the transcripts.

**How the ranges were found, and checked.** I found them by keyword matching on the transcripts and by reading the passages, and not with the search system, because the search would then grade itself. A second pass, which did not see my ranges, found them again from the questions alone. The two agreed within seconds on all 11. Where they differed slightly I took the union.

**Three findings.** Three items are only partly answered by the lecturer, so I marked them `partial`: the effect of raising or lowering a threshold (the trade-off is stated, not the direction), the axes of a curve (they are on the slide, not spoken), and why one property of a method is desirable (explained, but the lecturer argues little for it). For those, a good answer is a partial answer that says what is missing. One question on the second sheet is answered in the first lecture, so it tests search across lectures. And the transcript mishears some abbreviations (the abbreviation AUROC is often written as "rock"), which is why keyword matching alone would struggle on those questions.

**Reference answers.** For each item I keep a short answer written from the transcript only, and not from the internet, because the system must answer from the course. An answer from the web would measure whether the system knows the topic, and not whether it found what this lecturer said.

**Status.** The set is built, and `python -m src.evaluate` runs it (the first numbers are in decision 24). The file stays out of this repository (it is in an ignored folder), because the questions come from course material.

---

## 24. The first evaluation: how well the search finds the answer

**Problem.** Until now I had tried the search and the answer step on a handful of questions by hand. Before I change any setting I need a baseline, so that every later change can be judged by the same numbers.

**What I built.** `python -m src.evaluate` runs every question of the evaluation set (decision 23) through the real search over all three lectures and takes the best 10 chunks. A result is a *hit* when it comes from the right lecture and its time range shares at least one second with one of the question's answer ranges (touching at an edge does not count). The *rank* is the position of the first hit. From the ranks it reports hit@k (the share of questions that have a hit among the top k results) and mean reciprocal rank (MRR: the average of 1 divided by the rank, where a miss counts as 0), for all questions with an answer and for the fully and the partly answered ones separately. For the three questions the lectures do not cover, it prints the best score. With `--answers` it also writes an answer for every question with Gemini (one call each, paced under the free quota) and checks that those three are refused, that none of the others is, and that each answer cites a chunk from the right lecture and time range. Every run is saved with a snapshot of the settings (chunk size, which slide text is embedded, embedding model and token limit, number of results) in the ignored `data/eval/` folder, so two runs can be compared later. 22 tests cover the overlap rule, the rank finder, the metrics and the search on a tiny in-memory database. When I changed the overlap rule so that touching counted as overlapping, one test went red, as it should.

**Settings measured.** Chunks of about 350 words (maximum 450, one piece of overlap), the vision model's text as the slide text (`clean_text`), bge-m3 with a limit of 3,072 tokens, dense search only, 96 chunks.

**Results (2026-10-08).**

| Questions | Count | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| All with an answer | 11 | 0.91 (10 of 11) | 1.00 | 1.00 | 1.00 | 0.939 |
| Fully answered | 8 | 0.88 (7 of 8) | 1.00 | 1.00 | 1.00 | 0.917 |
| Partly answered | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.000 |

The one question that was not found at rank 1 had its hit at rank 3. Its answer is spread over about 17 minutes of lecture, and the best-scoring chunk came from just before that stretch.

The best result scored between 0.565 and 0.714 for the questions with an answer, and 0.403, 0.482 and 0.538 for the three without one. The two ranges nearly touch, so a cutoff on the score alone would be fragile.

The answer step, in one run at temperature 0: all three questions the lectures do not cover were refused, none of the 11 others was refused, and all 11 answers cite at least one chunk from the right lecture and time range. On a partly answered question, the answer gave the part the lecture covers and said that the rest is not covered.

**Reading the answers.** I read the 11 answers against my reference answers. Four of them contained details that my reference answers do not have, so for each of those I searched the five excerpts the model had been given. Three of the four were on the slides. The model reads the slide text, while I drafted the reference answers from the speech only (decision 23), so the reference answers were incomplete and the answers were not wrong. In the fourth, the statement is in the excerpts, but the lecturer makes it about a whole set of properties of the method, and the answer attaches it to the single property the question asked about and cites other excerpts than the one it comes from. So I found no invented claim and one citation that points at the wrong excerpt. This was a spot check (keyword search over the excerpts, then reading the passage around each match), not a full read of every excerpt.

**What this does not show.** It is a baseline, not proof that the search is good.
- With 11 questions, one question moves hit@1 by 9 points.
- Several answer ranges are many minutes long and a chunk is about 3 minutes, so a hit is easy to get. hit@3 and higher are already at 1.00, so they can only show a variant getting worse. hit@1 and MRR are the numbers to watch.
- The questions come from the exercise sheets and may be worded close to the lecture.
- The answer step was run once.
- A cited chunk in the right place does not prove that it supports every sentence it is attached to.

**Next.** These numbers are the baseline for the comparisons planned in decisions 15, 19 and 20 (which slide text is embedded, how much of it, chunk size, cutting at slide changes), and then for keyword search merged with this search. Because the numbers are already so high, the evaluation set needs more or harder questions before it can separate close variants. That is decision 25.

---

## 25. A harder evaluation set, and checking the answers claim by claim

**Problem.** The first set (decision 24) was too easy to learn from. Its best-3 results were already perfect, its questions come from exercise sheets and may be worded like the lectures, some answer ranges are many minutes long, it has no question on the code lecture, and it has only three questions the lectures do not cover.

**What I built.** A second private set in the same ignored folder: 36 questions with an answer and 4 without. By type: 7 paraphrased, 12 detail, 9 decoy, 3 cross-lecture decoy, 3 multi-hop, 2 partly covered, and the 4 not covered. The rules:
- Student wording, avoiding the slide's own terms where a natural paraphrase exists (the search must cope with different words for the same idea).
- Facts specific to the lecturer's own examples and demos, and not textbook definitions, so a model cannot answer from general knowledge.
- Tight answer ranges, usually 20 to 120 seconds.
- Every other place that gives the same answer is listed, and finding any one of them counts as a hit.
- A *decoy* question has a look-alike passage, in the same or another lecture, that shares the terms but does not answer it. A *multi-hop* question needs two places, and it is reported as "all places in the top k". A *not covered* question sounds on topic but is never answered.
- The code lecture is included for the first time (10 questions).

**How the answers were kept independent.** The questions were written from the transcripts and slide texts alone, not with the search, so the search does not grade itself. A second, separate pass then saw only the question texts and found the places and the answers on its own. A range was kept only where the two passes agreed, with the union of the two. An extra place was kept only if the second pass found it too (this removed ranges where text merely stays on screen while something else is discussed). Six places that only the second pass found were read by hand, and five were rejected as not answering the question. A not-covered question was kept only if the second pass also found no answer. Two questions meant as not covered turned out to be partly answered, and I turned them into partly answered questions.

**Results: finding the right place** (dense search, default settings, 36 questions with an answer):

| | Questions | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|---|---|
| First set (decision 24) | 11 | 0.91 | 1.00 | 1.00 | 1.00 | 0.939 |
| Hard set | 36 | 0.58 (21) | 0.83 | 0.92 | 0.94 | 0.711 |

By type, the first result is right for: paraphrased 0.71 of the time (MRR 0.857), detail 0.58 (0.697), decoy 0.44 (0.596), cross-lecture decoy 0.67 (0.733), partly covered 0.50 (0.583), and multi-hop 0.67 (0.833) when any one of its places counts. For the three multi-hop questions, all places are in the top 5 for two. On the code lecture the first result is right for 4 of 10. Two questions are not found in the top 10 at all, and for one of them the best-scoring result is the decoy passage it was built to attract.

**Results: the answer step** (40 answers, one run, temperature 0). All 4 questions the lectures do not cover were refused. 33 of the 36 other answers cite a chunk from the right place, and these are exactly the 33 questions where the right chunk is in the top 5, which is all the answer step reads. So answer quality is capped by hit@5 (0.92), and better search is the lever. The three others were not invented answers: the search missed twice, so the answer said the material does not cover it, and once the answer correctly said the lecture only names a test without explaining it, but set the "answerable" flag to false, so my total counts it as a wrong refusal. The best scores of the four not-covered questions (0.526 to 0.619) lie inside the range for the answerable ones (0.499 to 0.748), so a score cutoff cannot decide refusals.

**Results: claim by claim.** For each of the 36 answers, separate readers got only the question, the answer and the five excerpts the answer step had read, and judged every claim. There were 123 claims: 116 are supported by a cited excerpt, 5 are supported but by an excerpt that was not cited, 2 are overstated, and none is unsupported or contradicted. By answer: 30 fully faithful, 5 with a citation problem, 1 unfaithful. All five citation problems are the same pattern: the answer opens with a yes or no sentence that has no citation, and the next sentence repeats it with one. The one unfaithful answer states two things more strongly than its excerpts do (it adds a word the excerpt does not use, and it picks which use cases need a property when the slide only lists them). The three refusals were all judged true. Three answers are incomplete because the chunk that holds the missing part was not retrieved, not because of how they were written.

**A failure of my own tooling.** The first answer run stopped at question 3 of 40: Gemini answered "high demand" three times in a row, the answer step gives up after three quick tries, and the whole run was lost before anything was saved. The evaluation now waits 20, 40 and 80 seconds between tries and records an error for a question instead of stopping. Three new tests cover it, and removing the `continue` after a failed question makes one fail. 179 tests pass.

**What this does not show.**
- 36 questions is still small, and there are only 3 multi-hop questions.
- The places and reference answers come from two independent passes, but I have not checked each one by hand.
- The claim check is one pass per answer, and one call was a strictness call.
- It was one run of the answer step.

**Next.** Two small changes to the answer step: ask for a citation on every sentence, including the first (and warn in code when a sentence has none), and let an answer that gives only the covered part of a question set "answerable". Then the settings experiments of decisions 15, 19 and 20, judged by hit@1 and MRR on this set, then keyword search merged with this search, and reranking if the decoy questions stay weak, since a reranker is built for exactly that case.
