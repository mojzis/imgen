"""Per-model input shaping."""

import pytest

from imgen.models import resolve_model
from imgen.refs import data_uri


@pytest.fixture
def uri(ref_png):
    return data_uri(ref_png)


def test_nano_banana_puts_refs_in_image_input(ref_png, uri):
    inp = resolve_model("nano-banana").shape("p", "3:2", [ref_png], None)
    assert inp == {"prompt": "p", "aspect_ratio": "3:2", "image_input": [uri]}


def test_nano_banana_omits_image_input_without_refs():
    inp = resolve_model("nano-banana").shape("p", "1:1", [], None)
    assert "image_input" not in inp


def test_seed_is_forwarded_when_set():
    inp = resolve_model("nano-banana").shape("p", "1:1", [], 7)
    assert inp["seed"] == 7


def test_seed_is_omitted_when_none():
    inp = resolve_model("flux").shape("p", "1:1", [], None)
    assert "seed" not in inp


def test_flux_uses_first_ref_as_image_prompt(ref_png, uri, tmp_path):
    other = tmp_path / "other.png"
    other.write_bytes(b"x")
    inp = resolve_model("flux").shape("p", "1:1", [ref_png, other], None)
    assert inp["image_prompt"] == uri


def test_flux_sets_png_and_safety():
    inp = resolve_model("flux").shape("p", "2:3", [], None)
    assert inp == {
        "prompt": "p",
        "aspect_ratio": "2:3",
        "output_format": "png",
        "safety_tolerance": 2,
    }


def test_imagen_sets_safety_filter():
    inp = resolve_model("imagen-3").shape("p", "1:1", [], None)
    assert inp == {
        "prompt": "p",
        "aspect_ratio": "1:1",
        "output_format": "png",
        "safety_filter_level": "block_medium_and_above",
    }


def test_sdxl_lineart_maps_aspect_to_size():
    inp = resolve_model("sdxl-lineart").shape("p", "1:1", [], None)
    assert (inp["width"], inp["height"]) == (1024, 1024)
    assert inp["prompt"] == "p"
    assert inp["scheduler"] == "K_EULER"


def test_sdxl_lineart_landscape_aspect():
    inp = resolve_model("sdxl-lineart").shape("p", "3:2", [], None)
    assert (inp["width"], inp["height"]) == (1024, 683)


def test_sdxl_lineart_ignores_refs(ref_png):
    inp = resolve_model("sdxl-lineart").shape("p", "1:1", [ref_png], None)
    assert not any(k.startswith("image") for k in inp)


def test_sdxl_lineart_portrait_aspect():
    inp = resolve_model("sdxl-lineart").shape("p", "2:3", [], None)
    assert (inp["width"], inp["height"]) == (683, 1024)
