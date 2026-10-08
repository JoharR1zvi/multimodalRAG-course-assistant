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

# A chunk is sealed as soon as it holds this many words of speech...
CHUNK_TARGET_WORDS = 350

# ...and a speech piece is never added if it would push the chunk over this many words.
CHUNK_MAX_WORDS = 450

# How many speech pieces at the end of a chunk are repeated at the start of the next one,
# so an idea that was cut at the boundary appears in both chunks.
CHUNK_OVERLAP_SEGMENTS = 1

# A last chunk with fewer NEW words than this is merged into the chunk before it
# (if the result still fits under the maximum).
CHUNK_MIN_LAST_WORDS = 100

# Which version of the slide text goes into the embedded string: "slide_text" (raw OCR),
# "cleaned_text" (cleaned OCR) or "clean_text" (the vision model's reading).
# The three are compared in a later experiment (decision 15).
CHUNK_SLIDE_TEXT_SOURCE = os.environ.get("CHUNK_SLIDE_TEXT_SOURCE", "clean_text")

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
