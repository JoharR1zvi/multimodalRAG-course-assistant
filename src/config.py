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

# --- Keyframe extraction defaults (tuned on lecture_01; re-tune per lecture type) ---

# How much ONE pixel must change (0-255) to count as "changed"
KEYFRAME_DIFF_THRESHOLD = 40.0

# What fraction of the WHOLE frame must change to call it a new slide
KEYFRAME_CHANGE_AREA_THRESHOLD = 0.05

# Safety net: force a save after this many seconds with no detected change
KEYFRAME_MAX_GAP_SECONDS = 180.0

# How often (in seconds) we look at the video
KEYFRAME_INTERVAL_SECONDS = 5.0
