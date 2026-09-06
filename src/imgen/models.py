"""Model registry: which Replicate model, how to shape its input, what it costs."""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any

from imgen.errors import ModelError
from imgen.refs import data_uri

Shaper = Callable[[str, str, Sequence[pathlib.Path], int | None], dict[str, Any]]

DEFAULT_MODEL = "google/nano-banana"
# USD per image, Replicate list price: an estimate, not an invoice.
DEFAULT_PRICE = 0.04


def _with_seed(inp: dict[str, Any], seed: int | None) -> dict[str, Any]:
    if seed is not None:
        inp["seed"] = seed
    return inp


def shape_image_input(
    prompt: str, aspect: str, refs: Sequence[pathlib.Path], seed: int | None
) -> dict[str, Any]:
    """nano-banana / seedream family: refs go in as an ``image_input`` list."""
    inp: dict[str, Any] = {"prompt": prompt, "aspect_ratio": aspect}
    if refs:
        inp["image_input"] = [data_uri(p) for p in refs]
    return _with_seed(inp, seed)


def shape_flux(
    prompt: str, aspect: str, refs: Sequence[pathlib.Path], seed: int | None
) -> dict[str, Any]:
    """flux: one ``image_prompt``, png out, relaxed safety."""
    inp: dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect,
        "output_format": "png",
        "safety_tolerance": 2,
    }
    if refs:
        inp["image_prompt"] = data_uri(refs[0])
    return _with_seed(inp, seed)


def shape_imagen(
    prompt: str, aspect: str, refs: Sequence[pathlib.Path], seed: int | None
) -> dict[str, Any]:
    """imagen: text only, png out, medium safety filter."""
    inp: dict[str, Any] = {
        "prompt": prompt,
        "aspect_ratio": aspect,
        "output_format": "png",
        "safety_filter_level": "block_medium_and_above",
    }
    return _with_seed(inp, seed)


SDXL_LONG_SIDE = 1024


def aspect_to_size(aspect: str, long_side: int = SDXL_LONG_SIDE) -> tuple[int, int]:
    """``"3:2"`` -> ``(1024, 683)``: longer side fixed, shorter side scaled."""
    w, h = (int(x) for x in aspect.split(":"))
    ratio = Fraction(w, h)
    if ratio >= 1:
        return long_side, round(long_side / ratio)
    return round(long_side * ratio), long_side


def shape_sdxl_lineart(
    prompt: str, aspect: str, refs: Sequence[pathlib.Path], seed: int | None
) -> dict[str, Any]:
    """sdxl-lineart: explicit size, sampler settings, no image inputs."""
    width, height = aspect_to_size(aspect)
    inp: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": 20,
        "guidance_scale": 9,
        "scheduler": "K_EULER",
        "num_outputs": 1,
    }
    return _with_seed(inp, seed)


@dataclass(frozen=True)
class Model:
    """A Replicate model reference plus how to talk to it."""

    id: str
    shape: Shaper = shape_image_input
    price: float = DEFAULT_PRICE
    family: str = ""

    def with_id(self, model_id: str) -> Model:
        return replace(self, id=model_id)


# ``family`` is the substring matched against arbitrary model refs (versioned or not).
MODELS: tuple[Model, ...] = (
    Model("google/nano-banana", shape_image_input, 0.039, "nano-banana"),
    Model("bytedance/seedream-4", shape_image_input, 0.03, "seedream"),
    Model("black-forest-labs/flux-1.1-pro", shape_flux, 0.04, "flux"),
    Model("google/imagen-3", shape_imagen, 0.05, "imagen"),
    Model("cuuupid/sdxl-lineart", shape_sdxl_lineart, 0.01, "sdxl-lineart"),
)

_BY_ID = {m.id: m for m in MODELS}


def resolve_model(name: str) -> Model:
    """Exact id, alias (``"flux"``, ``"imagen-3"``), or a ref containing a known family.

    Aliases must match exactly, so ``"imagen-4"`` never silently becomes imagen-3.
    Unknown full refs are returned as-is with the image_input shape and default price.
    """
    if name in _BY_ID:
        return _BY_ID[name]
    if "/" not in name:
        for m in MODELS:
            if name in (m.family, m.id.partition("/")[2]):
                return m
        known = ", ".join(m.family for m in MODELS)
        raise ModelError(f"unknown model alias {name!r} (known: {known})")
    for m in MODELS:
        if m.family in name:
            return m.with_id(name)
    return Model(name)


def price(model: str) -> float:
    return resolve_model(model).price
