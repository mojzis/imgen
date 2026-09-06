"""Prompt templates with ``{placeholder}`` variables."""

from __future__ import annotations

import pathlib

from imgen.errors import PromptError


def render(template: str, /, **vars: str) -> str:
    """``str.format`` the template; missing vars and stray braces are PromptErrors."""
    try:
        return template.format(**vars).strip()
    except KeyError as e:
        raise PromptError(f"template needs variable {e.args[0]!r}") from e
    except (IndexError, ValueError) as e:
        raise PromptError(f"bad template: {e}") from e


def render_file(path: pathlib.Path, /, **vars: str) -> str:
    try:
        return render(path.read_text(), **vars)
    except PromptError as e:
        raise PromptError(f"{path}: {e}") from e
