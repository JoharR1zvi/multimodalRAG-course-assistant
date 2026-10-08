import os
from dotenv import load_dotenv

load_dotenv()

TESSERACT_CMD = os.environ.get("TESSERACT_CMD")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

# How many keyframes are sent to Gemini at the same time (the vision stage mostly waits
# for the network). This only helps when Gemini answers slowly; the real speed limit is
# the requests-per-minute quota below.
GEMINI_MAX_WORKERS = int(os.environ.get("GEMINI_MAX_WORKERS", "4"))

# Gemini's FREE tier allows 15 requests per minute per model. We stay just under it (14)
# by spacing requests evenly. With a paid key, raise this in .env (e.g. 100) to go faster.
GEMINI_REQUESTS_PER_MINUTE = int(os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "14"))

# Delete audio.wav as soon as transcript.json exists. The audio file is big (about 160 MB
# for a 90-minute lecture) and can be recreated from the video in seconds.
# Put DELETE_AUDIO_AFTER_TRANSCRIPT=false in .env to keep it.
DELETE_AUDIO_AFTER_TRANSCRIPT = os.environ.get("DELETE_AUDIO_AFTER_TRANSCRIPT", "true").strip().lower() != "false"

# --- Chunking defaults (chunking.py) ---

# All the lectures belong to one course for now. The id goes into every chunk.
COURSE_ID = os.environ.get("COURSE_ID", "course_01")

# The chunk sizes can be changed through the environment for experiments (decision 19), for
# example CHUNK_TARGET_WORDS=200 CHUNK_MAX_WORDS=260 CHUNK_MIN_LAST_WORDS=60.

# A chunk is sealed as soon as it holds this many words of speech...
CHUNK_TARGET_WORDS = int(os.environ.get("CHUNK_TARGET_WORDS", "350"))

# ...and a speech piece is never added if it would push the chunk over this many words.
CHUNK_MAX_WORDS = int(os.environ.get("CHUNK_MAX_WORDS", "450"))

# How many speech pieces at the end of a chunk are repeated at the start of the next one,
# so an idea that was cut at the boundary appears in both chunks.
CHUNK_OVERLAP_SEGMENTS = int(os.environ.get("CHUNK_OVERLAP_SEGMENTS", "1"))

# A last chunk with fewer NEW words than this is merged into the chunk before it
# (if the result still fits under the maximum).
CHUNK_MIN_LAST_WORDS = int(os.environ.get("CHUNK_MIN_LAST_WORDS", "100"))

# Cut at slide changes (an experiment, decision 19). Off by default: chunks are cut by length only.
# When on, a chunk that already holds CHUNK_SLIDE_AWARE_MIN_WORDS words of speech is sealed at the
# next real slide change (the slide title changes), instead of at CHUNK_TARGET_WORDS. If no slide
# change comes before CHUNK_MAX_WORDS, it is sealed there. Slides without a title give no signal,
# so the length rule is used for them.
CHUNK_SLIDE_AWARE = os.environ.get("CHUNK_SLIDE_AWARE", "false").strip().lower() == "true"
CHUNK_SLIDE_AWARE_MIN_WORDS = int(os.environ.get("CHUNK_SLIDE_AWARE_MIN_WORDS", "250"))

# Which version of the slide text goes into the embedded string: "slide_text" (raw OCR),
# "cleaned_text" (cleaned OCR), "clean_text" (the vision model's reading) or "none" (no slide
# text at all). The versions are compared in an experiment (decision 15).
CHUNK_SLIDE_TEXT_SOURCE = os.environ.get("CHUNK_SLIDE_TEXT_SOURCE", "clean_text")

# Whether the vision model's description of diagrams and code goes into the embedded string too.
# Put CHUNK_INCLUDE_DESCRIPTION=false in the environment to leave it out (an experiment, decision 20).
# The chunk keeps the description either way: only what is embedded and searched changes.
CHUNK_INCLUDE_DESCRIPTION = os.environ.get("CHUNK_INCLUDE_DESCRIPTION", "true").strip().lower() != "false"

# --- Retrieval signals (indexing.py, retriever.py) ---

# Which embedded texts the search uses, as a comma-separated list in the environment:
#   "full"   = the chunk's embed_text (slide text, description and speech, as set above)
#   "speech" = the speech of the chunk alone
# One signal is the plain search. With several (for example "speech,full") every signal is
# searched on its own and the lists are merged with reciprocal rank fusion (decision 27).
# Changing this needs a new database (python -m src.pipeline --all --force chunk, with another
# QDRANT_PATH), because the vectors stored are different.
# The default merges both: it scored best over the three question sets (decision 27).
RETRIEVAL_SIGNALS = []
for signal_name in os.environ.get("RETRIEVAL_SIGNALS", "speech,full").split(","):
    if signal_name.strip() != "":
        RETRIEVAL_SIGNALS.append(signal_name.strip())

# When signals are merged, each one contributes its best this-many results
FUSION_CANDIDATES = int(os.environ.get("FUSION_CANDIDATES", "20"))

# Reciprocal rank fusion: a result at rank r in a list adds 1 / (FUSION_RRF_K + r) to its score
FUSION_RRF_K = 60

# --- Embeddings (embedding_service.py) ---

# Which embedding model turns text into vectors: "local" (runs on this machine) or "gemini"
# (the Gemini API; not built yet). Swapping needs a new vector collection, see vector_store.py.
EMBEDDING_PROVIDER = os.environ.get("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

# Embeddings already computed are saved here, one small file per text (named by a hash of
# the model + the text), so running the indexing again never recomputes them.
EMBEDDING_CACHE_DIR = os.environ.get("EMBEDDING_CACHE_DIR", "data/cache/embeddings")

# How many texts go through the local model at once (a smaller number uses less GPU memory).
# Long texts need a small batch on a 6 GB card.
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "4"))

# Texts longer than this many tokens are cut off by the local model (the END is lost).
# A token is a piece of a word. Our embedded chunk texts (slide text + description + speech)
# measured up to 2549 tokens, so 3072 keeps all of them. bge-m3 itself accepts up to 8192.
EMBEDDING_MAX_TOKENS = int(os.environ.get("EMBEDDING_MAX_TOKENS", "3072"))

# --- Vector database (vector_store.py) ---

# Qdrant runs inside our own program and keeps its data in this folder (no server needed).
# Only one program can have the folder open at a time.
QDRANT_PATH = os.environ.get("QDRANT_PATH", "data/qdrant")

# Every embedding model gets its own collection: "<prefix>__<model name>", for example
# "course_chunks__bge-m3". Vectors from two different models must never be mixed.
QDRANT_COLLECTION_PREFIX = "course_chunks"

# --- OCR cleanup defaults (ocr_clean.py) ---

# Drop OCR words Tesseract is less sure about than this (0 to 1). Garbage like "oP" or "sso"
# scores 0 to 0.3, real slide words score 0.9+. But real CODE can score 0.4 to 0.6
# (import, hello.py), so the default is low. Can be raised per lecture in settings.json.
OCR_MIN_WORD_CONFIDENCE = 0.30

# A word is a "junk candidate" if it appears on at least this share of a lecture's keyframes
# (and on at least OCR_JUNK_MIN_SLIDES of them). The LLM then decides which candidates are
# software interface text and which are ordinary words.
OCR_JUNK_MIN_SLIDE_SHARE = 0.25
OCR_JUNK_MIN_SLIDES = 5

# Never send more than this many candidate words to the LLM
OCR_JUNK_MAX_CANDIDATES = 80

# --- Keyframe extraction defaults (tuned on lecture_01; re-tune per lecture type) ---

# How much ONE pixel must change (0-255) to count as "changed"
KEYFRAME_DIFF_THRESHOLD = 40.0

# What fraction of the WHOLE frame must change to call it a new slide
KEYFRAME_CHANGE_AREA_THRESHOLD = 0.05

# Safety net: force a save after this many seconds with no detected change
KEYFRAME_MAX_GAP_SECONDS = 180.0

# How often (in seconds) we look at the video
KEYFRAME_INTERVAL_SECONDS = 5.0
