"""Visual styles from yaml, and the glue that turns style + subject into a prompt."""

from __future__ import annotations

import pathlib
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from typing import Any

import yaml

from imgen.errors import StyleError


@dataclass(frozen=True)
class Style:
    """One visual language: style block, palette, mood, negatives, default aspect."""

    style: str
    palette: list[str] = field(default_factory=list)
    mood: str = ""
    negative: str = ""
    aspect_ratio: str = "1:1"
    coloring: str | None = None  # prompt used to derive a line-art page, if any
    model: str | None = None  # per-style model override

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Style:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise StyleError(f"unknown style key(s): {', '.join(sorted(unknown))}")
        if "style" not in data:
            raise StyleError("missing required key 'style'")
        return cls(**data)


def load_style(path: pathlib.Path) -> Style:
    """A single-style file (``theme.yaml``): top-level keys are the style fields."""
    try:
        return Style.from_dict(yaml.safe_load(path.read_text()) or {})
    except StyleError as e:
        raise StyleError(f"{path}: {e}") from e


def load_styles(path: pathlib.Path) -> dict[str, Style]:
    """A styles file (``styles.yaml``): mapping of style id -> style fields."""
    raw = yaml.safe_load(path.read_text()) or {}
    styles = {}
    for sid, data in raw.items():
        try:
            styles[sid] = Style.from_dict(data)
        except StyleError as e:
            raise StyleError(f"{path}: style {sid!r}: {e}") from e
    return styles


def compose(style: Style, subject: str, extra_negatives: Iterable[str] = ()) -> str:
    """style + ``Subject: ...`` + palette + mood + negatives, blank-line separated."""
    parts = [style.style.strip(), f"Subject: {subject.strip()}"]
    if style.palette:
        parts.append("Use this color palette: " + ", ".join(style.palette) + ".")
    if style.mood:
        parts.append(f"Mood: {style.mood}.")
    negatives = [style.negative, *extra_negatives]
    if neg := " ".join(n.strip() for n in negatives if n and n.strip()):
        parts.append(neg)
    return "\n\n".join(parts)
