import pytest

from imgen import ModelError
from imgen.models import DEFAULT_MODEL, DEFAULT_PRICE, price, resolve_model


def test_default_model_is_nano_banana():
    assert DEFAULT_MODEL == "google/nano-banana"


@pytest.mark.parametrize(
    ("name", "expected_id"),
    [
        ("nano-banana", "google/nano-banana"),
        ("google/nano-banana", "google/nano-banana"),
        ("seedream", "bytedance/seedream-4"),
        ("flux", "black-forest-labs/flux-1.1-pro"),
        ("imagen-3", "google/imagen-3"),
        ("imagen", "google/imagen-3"),
        ("flux-1.1-pro", "black-forest-labs/flux-1.1-pro"),
        ("sdxl-lineart", "cuuupid/sdxl-lineart"),
    ],
)
def test_resolve_by_alias_or_id(name, expected_id):
    assert resolve_model(name).id == expected_id


def test_resolve_unknown_model_keeps_id_and_default_shape():
    m = resolve_model("someone/new-model")
    assert m.id == "someone/new-model"
    assert m.shape("p", "1:1", [], None) == {"prompt": "p", "aspect_ratio": "1:1"}


def test_resolve_versioned_ref_matches_family():
    ref = "black-forest-labs/flux-1.1-pro:abc123"
    m = resolve_model(ref)
    assert m.id == ref
    assert m.price == price("flux")


def test_price_for_unknown_model_is_default():
    assert price("someone/new-model") == DEFAULT_PRICE


def test_price_by_substring():
    assert price("google/nano-banana") == pytest.approx(0.039)


@pytest.mark.parametrize("alias", ["imagen-4", "flux-schnell", "nano", "banana"])
def test_alias_must_match_exactly(alias):
    with pytest.raises(ModelError, match=alias):
        resolve_model(alias)
