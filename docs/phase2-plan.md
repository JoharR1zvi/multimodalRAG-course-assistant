# Phase 2 design: from knowledge objects to cited answers

Status: **built and measured, with evaluations on three question sets (decisions 24, 25, 27, 29 and 30).** The steps below are built up to cited answers and a small web page. The default search merges a speech search, a full-text search and a keyword search. A reranker is built and switched off.
Last updated: 2026-10-09.

## Where each step stands

| Step | What | Status |
|---|---|---|
| 2.0 | Carry all three versions of the slide text into the knowledge objects | Done |
| 2.1 | Chunking, `chunks.json`, checks in the exit check | Done (decision 19) |
| Evaluation set | Questions with known answer locations | Built, kept private (decision 23). `python -m src.evaluate` runs it. Baselines of the first search: hit@1 0.91, MRR 0.939 on the first set of 11 questions (decision 24), and hit@1 0.58, MRR 0.711 on a harder set of 36 (decision 25). A third set of 27 was added to check the tuning (decision 27). After tuning: 0.82 / 0.882, 0.75 / 0.844 and 0.70 / 0.815. With keyword search added (decision 29): 0.82 / 0.894, 0.89 / 0.931 and 0.78 / 0.861 |
| 2.2 | Embeddings behind one swappable interface | The local model (bge-m3) is built and used (decisions 12 and 20) |
| 2.3 | Vector store (embedded Qdrant) and indexing | Done (decision 21). Indexing is pipeline stage 10 |
| 2.4 | Retrieval | Done in three steps. Dense search with a baseline (`python -m src.search`, decision 24), then the settings comparison (which slide text, chunk size, cutting at slide changes) and a merge of two meaning searches by rank fusion (decision 27), then keyword search (BM25) as a third list in the same merge, which is now the default (decision 29). Over the 74 questions of the three sets the first result is right for 0.838 and the right place is in the top 3 for 0.973 |
| 2.5 | Reranking | Built and measured, switched off (`RERANK=true` to try it). A cross-encoder re-reads the best 20 candidates; neither way of using its order passed the rule of improving on every set (decision 30) |
| 2.6 | Cited answers | Done (`python -m src.ask`, decision 22). All 7 questions the lectures do not cover were refused, and 33 of the 36 harder answers cite the right place (the 33 where the right chunk was in the top 5). A claim-by-claim check found 116 of 123 claims supported by the cited excerpt and none invented (decision 25). Since then every sentence must carry a citation (answers with an uncited sentence went from 5 to 2 of 36), and the answer form has `full`, `partial` and `none` instead of yes or no (decision 26). On the search of decision 27 the answer step was run again on 70 questions: 218 of 227 claims are supported by the cited excerpt, and one answer picked one of two disagreeing sources without saying so (decision 28) |

How the first version differs from the plan below:

- **The embedding input limit.** The plan did not mention it. Chunks are sized in words, and the model counts tokens, so more than half of the embedded texts were being cut off. The limit is now 3,072 tokens and the limit is part of the cache key (decision 20).
- **The evaluation set lives in `data/eval/`,** an ignored folder, and not in `tests/retrieval_eval.json`, because the questions come from course material.
- **Commands:** `python -m src.search` and `python -m src.ask`. Indexing is the pipeline stage `index`, next to the new stage `chunk`.
- **A chunk also carries** `slide_titles`, and all three versions of the slide text (`slide_text`, `cleaned_text`, `clean_text`), so the retrieval experiments can swap them without rebuilding anything.
- **Citations are numbers, not times.** The answer model cites excerpt numbers, and the sources are looked up from the stored chunks (decision 22).

The diagrams in the README show the flow, and `docs/architecture.md` describes the stages.

## Goal

Phase 1 turns each lecture video into `knowledge_objects.json`: one object per slide, holding the slide text, any diagram description, everything said while the slide was on screen, and start/end times.

Phase 2 makes that searchable and answerable. A student asks a question, the system finds the right passages across lectures, and answers using only that material, citing the lecture, the slide, and the video timestamp.

The interface for Phase 2 is a command line, and a small web page on top of the same steps (`python -m src.api`).

## Architecture

Two separate flows. **Index time** runs once per lecture and is cached. **Query time** runs per question and must be fast.

```
INDEX TIME (per lecture)
knowledge_objects.json + alignment.json
   -> 2.1 chunk        -> chunks.json                  (inspectable)
   -> 2.2 embed        -> one vector per chunk
   -> 2.3 store        -> Qdrant (vector + metadata)

QUERY TIME (per question)
question
   -> embed the question (same model as the chunks)
   -> 2.4 retrieve     dense search (Qdrant) + keyword search (BM25), merged
   -> 2.5 rerank       cross-encoder re-scores the best ~20 (only if it measurably helps)
   -> 2.6 generate     LLM answers from the top chunks, with citations
```

## Terms used in this document

**Hierarchy.** Course > Week > Lecture > Topic > Segment > Chunk > Evidence. These are zoom levels, like an address. Only chunks are searched. The upper levels are labels used for filtering ("only lecture 2") and for citations. In practice the working levels now are Course > Lecture > Chunk. Week and Topic are optional labels that are not used.

**Segment vs chunk.** A segment is one slide's stretch of the lecture (a knowledge object). A chunk is the piece of text we embed and search. Chunks are cut by length, not by slide, so one chunk can cover several short slides.

**Embedding.** A list of numbers (a vector) that represents what a text means. Texts about similar ideas end up close together, so "how good is my classifier" can find a passage about accuracy and F1 without sharing any words. Closeness is measured with cosine similarity.

**Dense vs sparse.** A dense vector has a number in every position (an embedding, for example 1024 numbers). A sparse vector is mostly zeros, with a count per word (keyword search such as BM25). Dense finds meaning, sparse finds exact terms.

**Vector database (Qdrant).** A normal database finds exact matches. A vector database finds the nearest vectors to a query vector, quickly, and can filter on metadata at the same time. Qdrant runs embedded inside the Python process (no server), which is enough here.

## Data shapes (new schemas in `src/schemas/`)

`Chunk`: `chunk_id` (deterministic, e.g. `lecture_01_1420`, so re-indexing overwrites instead of duplicating), `course_id`, `lecture_id`, `start_timestamp`, `end_timestamp`, `slide_timestamps` (list of every slide the chunk covers), `image_paths` (list), `text` (the speech), `slide_text`, `slide_description`, `embed_text` (the string actually embedded), `prev_chunk_id`, `next_chunk_id`.

`SearchResult`: a `Chunk` plus `score` and the method that found it.

## 2.0 Carry all the slide text forward

Phase 1 produces three versions of each slide's text: the raw OCR text (`slide_text`), the cleaned OCR text (`cleaned_text`) and the vision model's reading (`clean_text`). Only the first is in `knowledge_objects.json` today. Before chunking I add the other two to the knowledge objects, with empty defaults so older files still load, and rebuild them for the three lectures. Then a chunk can carry all three, and the experiment in 2.4 can swap them (decision 15).

## 2.1 Chunking: `src/processing/chunking.py` -> `chunks.json`

Why chunk at all: a slide can hold several minutes of speech. Embedding all of it gives a blurry average of many ideas, and a hit returns a wall of text with a vague timestamp. A chunk should be one coherent stretch of speech.

- **Built from `alignment.json`**, not from the joined transcript in `knowledge_objects.json`. The joined string has lost per-sentence times, so a citation could only say "somewhere in this slide". The aligned segments keep exact start and end times.
- **Target size about 350 words**, hard maximum about 450, with one segment of overlap between neighbouring chunks so an idea cut at a boundary appears in both.
- **Consecutive slides merge** until a chunk reaches the target. Measured on lecture_01: median slide is 162 words, maximum 453, 26 of 64 slides are under 100 words, only 8 are over 350. Without merging, about 56 of 64 chunks would simply equal one slide and chunking would add nothing. Expect roughly 32 to 40 chunks for that lecture.
- **Blank and short slides join the next chunk**, because a title precedes its content. Section title slides ("Regression metrics") have no speech but carry the topic signal, so they must not be dropped.
- **What gets embedded (`embed_text`)**: slide text, then diagram description, then speech. Slide content is searched, not only speech, so a question phrased like the slide can match even when the speaker talks loosely about it.
- **Which slide text is embedded is decided by measurement.** Phase 1 now produces three versions of the slide text: the raw OCR text, a cleaned OCR text (`cleaned_text`), and the vision model's `clean_text`. The first retrieval experiment runs the same questions with each version in `embed_text` and compares hit@k (decisions 11 and 15). `cleaned_text` and `clean_text` are carried in `knowledge_objects.json` and in every chunk, and the experiment was run (decision 27).
- Same stage convention as Phase 1: skip when the output exists unless forced, build the data in memory, then write.

Checks: no chunk over the maximum; chunk times lie inside the span of the slides it lists; all speech present (apart from overlap); blank-slide titles appear in the next chunk.

## 2.2 Embeddings: `src/embeddings/embedding_service.py`

**Two providers are implemented and compared:**

- Local `BAAI/bge-m3` via `sentence-transformers`: free, runs locally, multilingual, 1024 dimensions, about 2 GB download.
- Gemini embeddings API: nothing to download, but subject to rate limits and sends the text off the machine.

Only the local model was built and used (decision 12).

**Swappable by design, without abstract classes:**
- Config holds `EMBEDDING_PROVIDER` (only "local" exists) and `EMBEDDING_MODEL`.
- The module exposes two public functions, `embed_texts(list)` and `embed_query(text)`, and picks a small private function per provider with a plain if/elif.
- Rules that make swapping safe: never hardcode the vector size (measure it by embedding one test string); name the Qdrant collection after the model (`course_chunks__bge-m3`) so vectors from different models never mix; keep any model-specific query or passage prefix in this one file; normalize vectors.
- Swapping a model means re-embedding from `chunks.json` into a new collection. That is cheap because `chunks.json` is the source of truth.
- Embeddings are cached on disk, keyed by a hash of the model, the token limit and the text, so a rerun does not embed anything twice.
- The GPU has 6 GB. Do not run Whisper and the local embedder at the same time.

Check: embed three sentences (two on the same topic, one unrelated); the related pair must score higher.

## 2.3 Vector store: `src/database/vector_store.py`

- Qdrant in embedded mode: `QdrantClient(path=...)`. No Docker, works on Windows, gitignored like other data.
- One collection per embedding model. Each point is a vector plus a payload with all chunk metadata, so results carry their own citations and can be filtered by lecture or time range.
- Functions: `create_collection`, `upsert_chunks` (idempotent thanks to deterministic ids), `search(vector, top_k, filters)`.
- Kept behind a small interface (`vector_store.py`), so the backend can be swapped.
- Scale note: a 13-lecture course is about 500 chunks, a few MB of vectors. A plain array would search instantly. Qdrant is used for learning, metadata filtering, and growth (more lectures and courses), not for speed.
- Indexing is added to the pipeline as stages after `knowledge`. The cache check for the DB stage is "this lecture already has points".

Check: point count equals chunk count; a raw search returns payloads with timestamps.

## 2.4 Retrieval: `src/retrieval/retriever.py`

1. **Dense only first.** Embed the question, search Qdrant, print the top 5 with score, lecture, `mm:ss` and a text preview. This is the baseline.
2. **Evaluation set before any tuning.** 15 to 20 real questions on lecture_01, each with the time range where the answer is spoken, stored in `tests/retrieval_eval.json`. The metric is hit@k: does a chunk overlapping the correct range appear in the top k? Every later change is judged by this number.
3. **Hybrid.** Add BM25 keyword search (`rank_bm25`) over the same chunks and merge both ranked lists with Reciprocal Rank Fusion: each result scores the sum of `1 / (60 + rank)` across the lists. Dense search is weak on exact terms such as formulas, acronyms and code identifiers, and keywords catch those. Kept only if hit@k improves. *Done (decision 29):* the keyword list reads the stored chunks, is merged with the two meaning lists by the same fusion, and improved the results on all three question sets.

## 2.5 Reranking (conditional): `src/retrieval/reranker.py`

A cross-encoder reads the question and a chunk together and scores the pair, which is more accurate than comparing two independently computed vectors, but slower because it runs once per pair. It re-scores the top ~20 results and keeps the best 5. It is added only if the evaluation set shows a clear gain. *Built and measured (decision 30), and left off:* it did not clear that bar. The model is `BAAI/bge-reranker-v2-m3`; `RERANK=true` turns it on, and `RERANK_MODE` chooses between replacing the order and blending it with the first search's order.

## 2.6 Generation: `src/generation/llm_service.py`

- Gemini answers from the top chunks, optionally with the slide images of the best two or three so it can read diagrams directly.
- Prompt rules: answer only from the numbered context; if the course does not cover it, say so; cite as `[lecture_01 @ 23:41]`.
- Citations are built from chunk metadata, never from the model's memory of timestamps.
- Output: the answer, then a Sources list (lecture, slide image, timestamp).

Checks: five answerable questions give correct cited answers; three unanswerable questions return "not in the course material"; each citation's timestamp really contains the claim.

## Storage decisions

- Videos stay on local disk (they are supplied by whoever runs the system, and only a path is needed).
- Measured on lecture_01: video 164 MB, `audio.wav` 162 MB, keyframes 8.7 MB, all JSON files together about 0.5 MB. The large regenerable file is `audio.wav`.
- `pipeline.py` deletes `audio.wav` once `transcript.json` exists (done, decision 16). The audio stage is skipped whenever the transcript exists, and forcing the transcript regenerates the audio first.
- A hosted database does not save meaningful disk (vectors are a few MB).

## OCR junk: generic cleanup

The recording interface (menu bar, tooltips) leaks into the OCR text. Fixed layout coordinates are rejected because they tie the pipeline to one recording style. Layers, cheapest first:

1. A layout-agnostic vision prompt that returns clean slide text and title, ignoring any application interface. It is the same Gemini call made per keyframe today.
2. A once-per-lecture junk list: count how many keyframes each word appears on, send the most frequent words in one request to an LLM and ask which are interface or recording junk versus real content, save the answer as an inspectable `junk_terms.json`, and apply it with plain code. Frequency finds the candidates and the model judges them. This handles both a legitimate word that happens to be frequent and OCR-corrupted menu words ("Fle Edt") that a pure frequency count would miss.
3. A per-word OCR confidence filter for random garbage tokens, which differ on every frame and never repeat.
4. Only if still noisy: drop words located in screen regions whose pixels never change across the whole video (data-driven, unlike a fixed crop).

**Status (2026-10-06):** layers 1 to 3 are built, and layer 4 has not been needed. The results on three lectures and the limits are in decision 11. The vision model's text and the cleaned OCR text are compared at retrieval time (decision 15).

## Risks

- Retrieval quality depends on Phase 1 quality: noisy OCR, missing titles and mid-sentence chunk edges all hurt. This is why the generalized vision prompt and the junk cleanup come first.
- Code lectures: embeddings of code are weaker and Tesseract loses indentation. Rely on the vision model's code transcription and keep keyword search in the mix.
- Citations must be real: timestamps come from metadata only.
- Model downloads and GPU memory (see 2.2).

## End-of-phase verification

On the three lectures (two slide lectures and one code lecture): hit@k and MRR are reported for the dense, the merged and the keyword search and for reranking (decisions 24 to 30); the answer step refuses questions the lectures do not cover and cites real places; re-running the index is a no-op; adding a lecture indexes only the new one.
