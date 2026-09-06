"""The one call that costs money: shape input, run the model, unwrap the output."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from typing import Any

import httpx
import replicate
from dotenv import find_dotenv, load_dotenv

from imgen.errors import GenerationError
from imgen.models import DEFAULT_MODEL, resolve_model

_run = replicate.run  # indirection so tests can swap the network call

TOKEN_VAR = "REPLICATE_API_TOKEN"  # noqa: S105 -- variable name, not a secret
URL_TIMEOUT = 120


def have_token() -> bool:
    """True when a Replicate token is set, loading ``.env`` from the cwd first."""
    if not os.getenv(TOKEN_VAR):
        load_dotenv(find_dotenv(usecwd=True))
    return bool(os.getenv(TOKEN_VAR))


def _fetch(url: str) -> bytes:
    try:
        resp = httpx.get(url, timeout=URL_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise GenerationError(f"could not fetch model output: {e}") from e
    return resp.content


def _unwrap(out: Any) -> bytes:
    if isinstance(out, list):
        if not out:
            raise GenerationError("the model returned no image")
        out = out[0]
    if hasattr(out, "read"):
        blob = out.read()
    elif isinstance(out, str):
        blob = _fetch(out)
    else:
        raise GenerationError(f"unexpected model output: {type(out).__name__}")
    if not blob:
        raise GenerationError("the model returned no image")
    return blob


def generate(  # noqa: PLR0913 -- keyword-only knobs
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    aspect: str = "1:1",
    refs: Sequence[pathlib.Path] = (),
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> bytes:
    """Generate one image and return its bytes.

    ``options`` is merged last into the shaped input, for model-specific knobs.
    """
    if not have_token():
        raise GenerationError(f"{TOKEN_VAR} is not set (put it in .env)")
    m = resolve_model(model)
    inp = m.shape(prompt, aspect, refs, seed)
    if options:
        inp.update(options)
    try:
        out = _run(m.id, input=inp)
    except Exception as e:
        raise GenerationError(str(e)) from e
    return _unwrap(out)


def prompt_note(
    prompt: str,
    *,
    model: str,
    aspect: str,
    refs: Sequence[pathlib.Path] = (),
    seed: int | None = None,
) -> str:
    """The ``.prompt.txt`` sidecar: header comments, blank line, the prompt."""
    head = [f"# model: {model}", f"# aspect: {aspect}"]
    if refs:
        head.append("# refs: " + ", ".join(r.name for r in refs))
    if seed is not None:
        head.append(f"# seed: {seed}")
    return "\n".join(head) + "\n\n" + prompt + "\n"


def generate_to(  # noqa: PLR0913 -- keyword-only knobs
    path: pathlib.Path,
    prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    aspect: str = "1:1",
    refs: Sequence[pathlib.Path] = (),
    seed: int | None = None,
    options: dict[str, Any] | None = None,
    force: bool = False,
) -> pathlib.Path:
    """Write the image to ``path`` plus a ``.prompt.txt`` sidecar next to it.

    Skips the call when ``path`` already exists, unless ``force``.
    """
    if path.exists() and not force:
        return path
    blob = generate(
        prompt, model=model, aspect=aspect, refs=refs, seed=seed, options=options
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    path.with_suffix(".prompt.txt").write_text(
        prompt_note(
            prompt, model=resolve_model(model).id, aspect=aspect, refs=refs, seed=seed
        )
    )
    return path
