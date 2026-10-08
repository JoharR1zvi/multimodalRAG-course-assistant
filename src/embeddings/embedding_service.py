# Turn text into embeddings (vectors): lists of numbers that represent what a text MEANS.
# Texts about similar ideas get vectors that are close together, so a question can find a
# passage that explains the same idea in different words.
#
# Two public functions:
#   embed_texts(list of text)  -> one vector per text (used for the chunks)
#   embed_query(text)          -> one vector (used for the student's question)
#
# The model is chosen in config.py (EMBEDDING_PROVIDER / EMBEDDING_MODEL), so swapping it
# touches only this file. Rules that make swapping safe:
#   - the vector size is never hardcoded, it is measured (get_vector_size)
#   - vectors are normalized to length 1, so "closeness" is a plain dot product
#   - any model-specific prefix for questions or passages belongs in THIS file
#     (bge-m3 needs none)
#   - the vector database collection is named after the model (see vector_store.py),
#     so vectors from two different models never get mixed

import hashlib
from pathlib import Path

import numpy as np

from src.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_CACHE_DIR,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MAX_TOKENS,
)

# The local model is big, so it is loaded once, the first time it is needed
_local_model = None

# The vector size is measured once and remembered
_vector_size = None


def get_model_slug() -> str:
    # "BAAI/bge-m3" -> "bge-m3": a short, file-name-safe name for the model
    name = EMBEDDING_MODEL.split("/")[-1]
    return name.lower()


def make_cache_path(text: str) -> Path:
    # Every text gets its own cache file. The name is a hash of the model name + the token
    # limit + the text, so changing the model, the limit (a lower limit cuts the text, which
    # changes the vector) or the text automatically means "not in the cache".
    key_text = EMBEDDING_MODEL + "\n" + str(EMBEDDING_MAX_TOKENS) + "\n" + text
    key = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    return Path(EMBEDDING_CACHE_DIR) / get_model_slug() / f"{key}.npy"


def load_local_model():
    global _local_model

    if _local_model is not None:
        return _local_model

    # Imported here, not at the top, because these libraries are heavy and only this
    # provider needs them
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Loading {EMBEDDING_MODEL} on {device} (the first time, this also downloads it)...")
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    model.max_seq_length = EMBEDDING_MAX_TOKENS

    # Half precision halves the GPU memory (about 1.1 GB instead of 2.2 GB)
    if device == "cuda":
        model.half()

    _local_model = model
    return _local_model


def count_over_limit(token_counts: list, limit: int) -> int:
    # How many texts are longer than the limit (the model would silently cut their ends)
    over = 0
    for token_count in token_counts:
        if token_count > limit:
            over = over + 1
    return over


def embed_with_local_model(texts: list) -> np.ndarray:
    model = load_local_model()

    # Never let text be cut off silently: count the tokens first and warn if any text is too long
    token_counts = []
    for text in texts:
        token_ids = model.tokenizer(text, add_special_tokens=True)["input_ids"]
        token_counts.append(len(token_ids))

    too_long = count_over_limit(token_counts, EMBEDDING_MAX_TOKENS)
    if too_long > 0:
        print(f"WARNING: {too_long} of {len(texts)} texts are longer than {EMBEDDING_MAX_TOKENS} tokens (longest: {max(token_counts)}). Their ends are cut off.")

    vectors = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return vectors.astype(np.float32)


def embed_with_provider(texts: list) -> np.ndarray:
    # Picks the right function for the configured provider (a plain if/elif, no classes)
    if EMBEDDING_PROVIDER == "local":
        return embed_with_local_model(texts)

    if EMBEDDING_PROVIDER == "gemini":
        raise NotImplementedError("The Gemini embedding provider is not built yet. Use EMBEDDING_PROVIDER=local.")

    raise ValueError(f"Unknown EMBEDDING_PROVIDER {EMBEDDING_PROVIDER!r}. Use 'local' or 'gemini'.")


def get_vector_size() -> int:
    # How many numbers one vector has. Measured by embedding one test string, never hardcoded,
    # because every model has its own size.
    global _vector_size

    if _vector_size is None:
        test_vector = embed_with_provider(["vector size test"])
        _vector_size = int(test_vector.shape[1])

    return _vector_size


def embed_texts(texts: list) -> np.ndarray:
    # One vector per text, in the same order as the texts. Shape: (number of texts, vector size).
    # Texts that were embedded before are read from the cache; only the rest go through the model.

    if len(texts) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    vectors = [None] * len(texts)

    # texts the model still has to embed. A text that appears twice is embedded only once.
    missing_texts = []
    positions_of_missing = {}

    for i, text in enumerate(texts):
        cache_path = make_cache_path(text)

        if cache_path.exists():
            vectors[i] = np.load(cache_path)
            continue

        if text not in positions_of_missing:
            positions_of_missing[text] = []
            missing_texts.append(text)
        positions_of_missing[text].append(i)

    if len(missing_texts) > 0:
        new_vectors = embed_with_provider(missing_texts)

        for j, text in enumerate(missing_texts):
            vector = new_vectors[j]

            # Save it, so the next run finds it in the cache
            cache_path = make_cache_path(text)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, vector)

            for position in positions_of_missing[text]:
                vectors[position] = vector

    return np.stack(vectors)


def embed_query(text: str) -> np.ndarray:
    # The vector for one question. Questions are not cached (they are rarely repeated).
    # bge-m3 needs no special prefix on questions. A model that did (e.g. "query: ")
    # would get it added here and nowhere else.
    return embed_with_provider([text])[0]
