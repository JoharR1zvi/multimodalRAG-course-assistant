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
