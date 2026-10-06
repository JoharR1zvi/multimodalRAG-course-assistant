# conftest.py is read by pytest BEFORE it loads any test file.
# It is the place for setup that every test needs.

import os

# vision.py creates the Gemini client the moment it is imported, and that needs SOME key.
# Our tests never talk to Gemini, so a fake key is enough. Setting it here also means the
# real key from .env is never loaded during tests (an environment variable that is already
# set wins over .env). So if a test ever tries to call Gemini by accident, it simply fails
# with an invalid-key error instead of quietly spending quota.
os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")
