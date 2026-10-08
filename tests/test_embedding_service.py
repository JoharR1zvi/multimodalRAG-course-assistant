# Tests for embedding_service.py: the cache and the provider switch.
#
# The real model is big (2 GB), so these tests replace it with a tiny fake "model" that gives
# every text a fixed vector and counts how many texts it was asked to embed. That is enough
# to test everything this file is responsible for.

import numpy as np
import pytest

from src.embeddings import embedding_service


class FakeModel:
    # Pretends to be the embedding model. The vector of a text is [its length, 1.0].
    def __init__(self):
        self.texts_it_was_asked_for = []

    def __call__(self, texts):
        vectors = []
        for text in texts:
            self.texts_it_was_asked_for.append(text)
            vectors.append([float(len(text)), 1.0])
        return np.array(vectors, dtype=np.float32)


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    # Use a temporary cache folder, and swap the real model for the fake one
    fake = FakeModel()
    monkeypatch.setattr(embedding_service, "EMBEDDING_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(embedding_service, "embed_with_provider", fake)
    monkeypatch.setattr(embedding_service, "_vector_size", None)
    return fake


def test_one_vector_comes_back_per_text_in_the_same_order(fake_model):
    vectors = embedding_service.embed_texts(["aa", "bbbb", "c"])

    assert vectors.shape == (3, 2)
    assert vectors[0][0] == 2.0
    assert vectors[1][0] == 4.0
    assert vectors[2][0] == 1.0


def test_texts_already_embedded_are_read_from_the_cache_not_the_model(fake_model):
    embedding_service.embed_texts(["aa", "bbbb"])
    assert len(fake_model.texts_it_was_asked_for) == 2

    # second run: nothing new, so the model must not be asked again
    again = embedding_service.embed_texts(["aa", "bbbb"])

    assert len(fake_model.texts_it_was_asked_for) == 2
    assert again[1][0] == 4.0


def test_only_the_new_texts_go_to_the_model_and_the_order_is_kept(fake_model):
    embedding_service.embed_texts(["aa"])

    vectors = embedding_service.embed_texts(["aa", "bbbb", "cc"])

    # "aa" came from the cache; "bbbb" and "cc" were new
    assert fake_model.texts_it_was_asked_for == ["aa", "bbbb", "cc"]
    assert vectors[0][0] == 2.0
    assert vectors[1][0] == 4.0
    assert vectors[2][0] == 2.0


def test_a_text_that_appears_twice_is_embedded_once(fake_model):
    vectors = embedding_service.embed_texts(["same", "other", "same"])

    assert fake_model.texts_it_was_asked_for == ["same", "other"]
    assert vectors.shape == (3, 2)
    assert vectors[0][0] == vectors[2][0]


def test_the_cache_name_depends_on_the_model_and_the_text(fake_model, monkeypatch):
    path_a = embedding_service.make_cache_path("hello")
    path_b = embedding_service.make_cache_path("hello!")
    assert path_a != path_b

    monkeypatch.setattr(embedding_service, "EMBEDDING_MODEL", "someone/other-model")
    path_c = embedding_service.make_cache_path("hello")
    assert path_a != path_c


def test_the_cache_name_depends_on_the_token_limit(fake_model, monkeypatch):
    # A lower limit cuts the text shorter, which changes the vector: old vectors must not be reused
    path_with_big_limit = embedding_service.make_cache_path("hello")

    monkeypatch.setattr(embedding_service, "EMBEDDING_MAX_TOKENS", 512)
    path_with_small_limit = embedding_service.make_cache_path("hello")

    assert path_with_big_limit != path_with_small_limit


def test_texts_over_the_token_limit_are_counted():
    assert embedding_service.count_over_limit([100, 1024, 1025, 3000], 1024) == 2
    assert embedding_service.count_over_limit([], 1024) == 0
    assert embedding_service.count_over_limit([5, 6], 1024) == 0


def test_no_texts_means_an_empty_result(fake_model):
    vectors = embedding_service.embed_texts([])

    assert len(vectors) == 0


def test_the_vector_size_is_measured_not_hardcoded(fake_model):
    assert embedding_service.get_vector_size() == 2


def test_a_query_is_embedded_without_the_cache(fake_model, tmp_path):
    vector = embedding_service.embed_query("a question")

    assert vector[0] == float(len("a question"))
    # nothing was saved for a question
    cache_folder = tmp_path / "cache"
    assert not cache_folder.exists()


def test_the_model_slug_is_a_short_file_name_safe_name():
    assert embedding_service.get_model_slug() == "bge-m3"


def test_an_unknown_provider_is_refused(monkeypatch):
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "nonsense")

    with pytest.raises(ValueError):
        embedding_service.embed_with_provider(["x"])


def test_the_gemini_provider_says_it_is_not_built_yet(monkeypatch):
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "gemini")

    with pytest.raises(NotImplementedError):
        embedding_service.embed_with_provider(["x"])
