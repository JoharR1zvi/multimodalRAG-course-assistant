# conftest.py is read by pytest BEFORE it loads any test file.
# It is the place for setup that every test needs.

import os

# vision.py creates the Gemini client the moment it is imported, and that needs SOME key.
# Our tests never talk to Gemini, so a fake key is enough. Setting it here also means the
# real key from .env is never loaded during tests (an environment variable that is already
# set wins over .env). So if a test ever tries to call Gemini by accident, it simply fails
# with an invalid-key error instead of quietly spending quota.
os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-tests")

# The real default is to search two stored vectors and merge them (see src/config.py). Most tests
# build a small database with ONE vector per chunk, so they ask for the plain layout here, and the
# tests of the two-vector search pass their own `signals`. This also means the tests give the same
# result whatever RETRIEVAL_SIGNALS the person running them has set. (Plain assignment, not
# setdefault, on purpose: a value left over in the shell must not change the tests.)
os.environ["RETRIEVAL_SIGNALS"] = "full"

# Reranking loads a 2 GB model, so the tests keep it off, whatever the shell says. The tests of the
# reranker pass their own fake model.
os.environ["RERANK"] = "false"
